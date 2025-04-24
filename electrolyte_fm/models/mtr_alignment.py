from itertools import chain
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.loggers import WandbLogger

from ..utils.metrics import (
    masked_loss,
)
from ..utils.transforms import IdentityTransform, PowerTransform, Standardize
from .model_utils import DeepSpeedMixin, LoggingMixin
from .prediction_task_head import PredictionTaskHead


class MTRAlignment(pl.LightningModule, DeepSpeedMixin, LoggingMixin):
    """
    PyTorch Lightning module for multiple target regression (MTR) pre-traning.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        output_size: int = 210,
        dropout: float = 0.2,
        vocab_size: Optional[int] = None,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.encoder_ckpt = encoder_ckpt
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

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

        self.task_network = PredictionTaskHead(
            embed_dim=self.encoder.config.hidden_size,
            output_size=output_size,
            dropout=dropout,
        )

        self.lossfn = torch.nn.MSELoss(reduction="none")
        transform = transform or "standardize"

        # Init Transform
        if transform == "standardize":
            self.transform = Standardize(output_size)
        elif transform == "power_transform":
            self.transform = PowerTransform(output_size)
        else:
            self.transform = IdentityTransform()
        self.transform.eval()

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def on_fit_start(self):
        """Standardized training data"""
        if not isinstance(self.transform, IdentityTransform):
            state = None
            if self.global_rank == 0:
                assert self.trainer.datamodule.target_dataset is not None
                ds = self.trainer.datamodule.target_dataset
                state = self.transform.fit(ds)

            state = self.trainer.strategy.broadcast(state)
            self.transform.load_state_dict(state)

    # type: ignore[override]
    def forward(self, batch, transform=True, **kwargs):
        hs = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            **kwargs,
        ).last_hidden_state

        pred_unscaled = self.task_network(hs)
        if transform:
            return self.transform.forward(pred_unscaled)
        return pred_unscaled

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds = self.forward(batch, transform=False)
        target = batch["target"]
        print(f"target {target}")
        target = self.transform.inverse(target)
        print(f"transformed target {target}")
        loss = masked_loss(self.lossfn, preds, target, batch["target_mask"])
        preds = self.transform.forward(preds)
        return preds, loss

    def training_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        print(f"loss {loss}")
        exit()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, *args):
        hs = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).last_hidden_state
        embedding = hs[:, 0, :]

        preds = self.task_network(hs)
        preds = self.transform.forward(preds)

        out = {"embedding": embedding, "prediction": preds}

        out["target"] = batch["target"]

        return out

    def configure_optimizers(self):
        learnable_params = chain(
            self.task_network.parameters(), self.encoder.parameters()
        )
        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
