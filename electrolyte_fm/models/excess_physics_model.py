from itertools import chain
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger

from ..utils.metrics import (
    get_metrics,
    masked_metric_update,
    bootstrap_collection,
    masked_loss,
)
from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin, LoggingMixin
from .normalize import AbstractNormalizer
from .polynomial_task_head import PolynomialHead, FusionStrategy


class ExcessPhysicsModel(LightningModule, DeepSpeedMixin, LoggingMixin):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        include_linear_mixing: bool = True,
        tokenizer: Optional[str] = None,
        vocab_size: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: Optional[int] = 4,
        n_components: int = 2,
        polynomial_order: int = 4,
        fusion: str | FusionStrategy = FusionStrategy.ATTENTION,
        basis: str | PolynomialHead = PolynomialHead.RK,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        metrics: List[str] = ["mae", "rmse", "mape"],
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder_ckpt = encoder_ckpt
        self.include_linear_mixing = include_linear_mixing
        self.tokenizer = load_tokenizer(tokenizer or encoder_ckpt)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        self.n_components = n_components
        self.temperature_normalization = (273, 400)
        self.basis = PolynomialHead(basis)
        transform or "identity"
        self.transform = AbstractNormalizer.get(transform, 1).eval()

        # Load Encoder Model
        if Path(encoder_ckpt).exists():
            self.encoder = DeepSpeedMixin.load(encoder_ckpt).get_encoder()
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(
                encoder_ckpt,
                trust_remote_code=True,
            )

        self.hidden_size = self.encoder.config.hidden_size
        # Validate the vocab size
        if vocab_size is not None:
            if hasattr(self.encoder, "config") and hasattr(
                self.encoder.config, "vocab_size"
            ):
                assert (
                    self.encoder.config.vocab_size == vocab_size
                ), f"Expected vocab size to match. got {self.encoder.config.vocab_size} and {vocab_size}"

        task_head_args = {
            "fusion": fusion,
            "embed_dim": self.hidden_size,
            "polynomial_order": polynomial_order,
            "n_components": n_components,
            "num_heads": num_heads,
            "include_linear_mixing": self.include_linear_mixing,
        }

        self.task_network = self.basis.get_class()(**task_head_args)
        self.lossfn = torch.nn.MSELoss(reduction="mean")

        metrics = get_metrics(
            metrics,
            "regression",
            num_outputs=1,
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def on_fit_start(self):
        """Standardized training data"""
        if self.global_rank == 0:
            assert self.trainer.datamodule.target_dataset is not None
            ds = self.trainer.datamodule.target_dataset
            state = self.transform.fit(ds)

        state = self.trainer.strategy.broadcast(state)
        self.transform.load_state_dict(state)

    def forward(self, batch, transform=True, **kwargs):  # type: ignore[override]
        mn, mx = self.temperature_normalization
        temperature = (batch["temperature"] - mn) / (mx - mn)
        batch["temperature"] = temperature
        for i in range(self.n_components):
            enc_out = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=False,
            )

            token_seq = enc_out.last_hidden_state  # (B, L_i, d)
            padmask = batch[f"attention_mask_{i}"] == 0  # (B, L_i)  bool

            # save for cross-attention fusion
            batch[f"tokens_{i}"] = token_seq.float()
            batch[f"padmask_{i}"] = padmask

            # mean-pool tokens_i: single-molecule embedding
            pooled = token_seq.mean(axis=1)
            batch[f"embedding_{i}"] = pooled

        #  property prediction
        pred = self.task_network(batch)  # (B, 1)
        if transform:
            pred = self.transform.forward(pred)
        return pred  # [batch_size, 1]

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds = self.forward(batch, transform=False)
        target = batch["target"]
        target = self.transform.inverse(target)
        loss = self.lossfn(preds, target)
        preds = self.transform.forward(preds)
        return preds, loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        masked_metric_update(
            self.train_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            self.train_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        masked_metric_update(
            self.val_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        masked_metric_update(
            self.test_metrics,
            preds.to(dtype=torch.float32),
            batch["target"].to(dtype=torch.float32),
            batch["target_mask"],
        )
        return loss

    def on_test_epoch_end(self):
        self.log_dict(
            self.test_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.test_metrics.reset()

    def configure_optimizers(self):
        learnable_params = self.task_network.parameters()
        if not self.freeze_encoder:
            learnable_params = chain(learnable_params, self.encoder.parameters())

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer


class MultiTargetExcessPhysicsModel(ExcessPhysicsModel):
    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        include_linear_mixing: List[bool] = [True, True],
        tokenizer: Optional[str] = None,
        vocab_size: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: Optional[int] = 4,
        n_components: int = 2,
        n_targets: int = 2,
        polynomial_order: int = 4,
        fusion: str | FusionStrategy = FusionStrategy.ATTENTION,
        basis: str | PolynomialHead = PolynomialHead.RK,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        metrics: List[str] = ["mae-channel"],
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(encoder_ckpt=encoder_ckpt, basis=basis)

        self.freeze_encoder = freeze_encoder
        self.encoder_ckpt = encoder_ckpt
        self.tokenizer = load_tokenizer(tokenizer or encoder_ckpt)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        self.n_components = n_components
        self.num_heads = num_heads
        self.include_linear_mixing = include_linear_mixing
        self.temperature_normalization = (273, 400)

        transform = transform or "identity"
        self.transform = AbstractNormalizer.get(transform, n_targets).eval()

        # Load Encoder Model
        if Path(encoder_ckpt).exists():
            self.encoder = DeepSpeedMixin.load(encoder_ckpt).get_encoder()
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(
                encoder_ckpt,
                trust_remote_code=True,
            )

        self.hidden_size = self.encoder.config.hidden_size
        # Validate the vocab size
        if vocab_size is not None:
            if hasattr(self.encoder, "config") and hasattr(
                self.encoder.config, "vocab_size"
            ):
                assert (
                    self.encoder.config.vocab_size == vocab_size
                ), f"Expected vocab size to match. got {self.encoder.config.vocab_size} and {vocab_size}"

        task_head_args = [
            {
                "fusion": fusion,
                "embed_dim": self.hidden_size,
                "polynomial_order": polynomial_order,
                "n_components": n_components,
                "num_heads": num_heads,
                "include_linear_mixing": i,
            }
            for i in self.include_linear_mixing
        ]

        self.task_networks = nn.ModuleList(
            [self.basis.get_class()(**t) for t in task_head_args]
        )
        self.lossfn = torch.nn.MSELoss(reduction="none")

        metrics = get_metrics(
            metrics,
            "regression",
            num_outputs=len(self.task_networks),
        )
        metrics = bootstrap_collection(metrics, num_bootstraps=100)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds = self.forward(batch, transform=False)
        target = batch["target"]
        target = self.transform.inverse(target)
        loss = masked_loss(self.lossfn, preds, target, batch["target_mask"])
        preds = self.transform.forward(preds)
        return preds, loss

    def task_network(self, batch):
        pred = torch.hstack(tuple(t(batch) for t in self.task_networks))
        return pred

    def configure_optimizers(self):
        learnable_params = self.task_networks[0].parameters()
        for task_net in self.task_networks[1:]:
            learnable_params = chain(learnable_params, task_net.parameters())
        if not self.freeze_encoder:
            learnable_params = chain(learnable_params, self.encoder.parameters())

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
