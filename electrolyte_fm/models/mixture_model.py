from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from ..utils.metrics import (
    bootstrap_collection,
    get_metrics,
    masked_loss,
    masked_metric_update,
)
from .model_utils import DeepSpeedMixin
from .normalize import AbstractNormalizer
from .prediction_task_head import PredictionTaskHead
from .physics_task_heads import ArrheniusTaskHead, VFTTaskHead
from enum import Enum
from lightning.pytorch.loggers import WandbLogger


class TemperatureCondition(Enum):
    """Enumeration of supported embedding fusion strategies."""

    CONCAT = "concat"
    ARRHENIUS = "arrhenius"
    VFT = "vft"
    NONE = False


class MixtureModel(LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for mixture property regression.
    """

    def __init__(
        self,
        output_size: int,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        vocab_size: Optional[int] = None,
        metrics: List[str] = ["mae", "mae-channel", "r2", "r2-channel", "rmse-channel"],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str | list[str]] = None,
        tokenizer: Optional[str] = None,
        bootstrap: Union[bool, int] = False,
        target_columns: Optional[List[str]] = None,
        n_components: int = 5,
        temperature: str | bool | TemperatureCondition = False,
    ) -> None:
        super().__init__()

        self.output_size = output_size
        self.dropout = dropout
        self.encoder_ckpt = encoder_ckpt
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.freeze_encoder = freeze_encoder
        self.n_components = n_components
        self.temperature = TemperatureCondition(temperature)

        self.lossfn = torch.nn.MSELoss(reduction="none")
        self.transform = AbstractNormalizer.get(transform, self.output_size).eval()

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

        self.save_hyperparameters()

        if self.temperature is TemperatureCondition.NONE:
            self.task_network = PredictionTaskHead(
                embed_dim=self.encoder.config.hidden_size,
                output_size=output_size,
                dropout=dropout,
            )
        elif self.temperature is TemperatureCondition.ARRHENIUS:
            self.task_network = ArrheniusTaskHead(
                embed_dim=self.encoder.config.hidden_size
            )
        elif self.temperature is TemperatureCondition.VFT:
            self.task_network = VFTTaskHead(embed_dim=self.encoder.config.hidden_size)
        elif self.temperature is TemperatureCondition.CONCAT:
            # temperature concat increase emb size passed
            self.task_network = PredictionTaskHead(
                embed_dim=self.encoder.config.hidden_size + 1,
                output_size=output_size,
                dropout=dropout,
            )

        metrics = get_metrics(
            metrics,
            "regression",
            num_outputs=output_size,
            target_channels=target_columns,
        )
        metrics = bootstrap_collection(metrics, num_bootstraps=100)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def on_fit_start(self):
        """Standardized training data"""
        state = None
        if self.global_rank == 0:
            assert self.trainer.datamodule.target_dataset is not None
            ds = self.trainer.datamodule.target_dataset
            state = self.transform.fit(ds)

        state = self.trainer.strategy.broadcast(state)
        self.transform.load_state_dict(state)

    def forward(self, batch, transform=True, **kwargs):  # type: ignore[override]
        mix_embedding = []
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)
            composition = batch[f"composition_{i}"].view(-1, 1)
            embedding_scaled = embedding * composition  # Shape: [batch, embedding]
            mix_embedding.append(embedding_scaled)
        # Shape: [batch, n_components, embedding]
        mix_embedding = torch.stack(mix_embedding, dim=1).sum(axis=1)

        if self.temperature is TemperatureCondition.CONCAT:
            mix_embedding = torch.hstack(
                (batch["temperature"].view(-1, 1) / 400, mix_embedding)
            )
        pred_unscaled = self.task_network(mix_embedding)
        if transform:
            return self.transform.forward(pred_unscaled), mix_embedding
        return pred_unscaled, mix_embedding

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds, mix_embedding = self.forward(batch, transform=False)
        target = batch["target"]
        target = self.transform.inverse(target)
        loss = masked_loss(self.lossfn, preds, target, batch["target_mask"])

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

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

    def predict_step(self, batch, *args):
        mix_embedding = None
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)

            embedding = torch.stack(
                [
                    torch.mul(embedding[j, :], batch[f"composition_{i}"][j])
                    for j in range(embedding.shape[0])
                ]
            )
            if mix_embedding is None:
                mix_embedding = embedding
            else:
                mix_embedding += embedding

        pred_unscaled = self.task_network(mix_embedding, batch["temperature"])
        preds = self.transform.forward(pred_unscaled)

        out = {"embedding": embedding, "prediction": preds}

        for key in ["target", "is_oov"]:
            if key in batch.keys():
                out[key] = batch[key]

        return out

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
