from itertools import chain
from pathlib import Path
from typing import Optional

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanAbsoluteError

from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin, LoggingMixin
from .normalize import Standardize
from .polynomial_task_head import (
    ChebyshevPredictionTaskHead,
    LegendrePredictionTaskHead,
    RKPredictionTaskHead,
)


class ExcessPhysicsModel(LightningModule, DeepSpeedMixin, LoggingMixin):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        tokenizer: Optional[str] = None,
        vocab_size: Optional[int] = None,
        hidden_size: int = 768,
        dropout: float = 0.1,
        n_components: int = 2,
        polynomial_order: int = 4,
        basis: str = "rk",
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder_ckpt = encoder_ckpt
        self.tokenizer = load_tokenizer(tokenizer or encoder_ckpt)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        self.n_components = n_components
        self.temperature_normalization = (273, 400)

        # Load Encoder Model
        if Path(encoder_ckpt).exists():
            self.encoder = DeepSpeedMixin.load(encoder_ckpt).get_encoder()
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(
                encoder_ckpt,
                trust_remote_code=True,
            )

        # Validate the vocab size
        if vocab_size is not None:
            if hasattr(self.encoder, "config") and hasattr(
                self.encoder.config, "vocab_size"
            ):
                assert (
                    self.encoder.config.vocab_size == vocab_size
                ), f"Expected vocab size to match. got {self.encoder.config.vocab_size} and {vocab_size}"

        if basis == "rk":
            task_network = RKPredictionTaskHead(
                embed_dim=hidden_size,
                polynomial_order=polynomial_order,
                n_components=n_components,
            )
        elif basis == "legendre":
            task_network = LegendrePredictionTaskHead(
                embed_dim=hidden_size,
                polynomial_order=polynomial_order,
                n_components=n_components,
            )
        elif basis == "chebyshev":
            task_network = ChebyshevPredictionTaskHead(
                embed_dim=hidden_size,
                polynomial_order=polynomial_order,
                n_components=n_components,
            )

        self.task_network = task_network
        self.lossfn = torch.nn.MSELoss(reduction="mean")
        self.metric = MeanAbsoluteError()
        self.transform = Standardize(1)

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
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)
            embedding = torch.hstack((temperature.view(-1, 1), embedding))
            batch[f"embedding_{i}"] = embedding.float()
        pred_unscaled = self.task_network(batch).flatten()
        if transform:
            return self.transform.forward(pred_unscaled)
        return pred_unscaled

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
        self.metric.update(preds, batch["target"])
        self.log(
            "train/mae",
            value=self.metric.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.metric.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        self.metric.update(preds, batch["target"])
        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val/mae",
            value=self.metric.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.metric.reset()

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
        self.log("test/mae", self.metric)
        return loss

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
