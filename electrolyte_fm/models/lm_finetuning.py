from itertools import chain
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torchmetrics import MetricCollection

from ..utils.metrics import (
    OOVMetric,
    get_metric,
    masked_loss,
    masked_metric_forward,
)
from .model_utils import record_summary_stats
from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin
from .prediction_task_head import PredictionTaskHead


class Standardize(torch.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        assert (std > 0).all() and mean.isfinite().all()
        self.mean = mean
        self.std = std.maximum(torch.tensor(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.std * x) + self.mean

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class LMFinetuning(pl.LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on multiple tasks.
    """

    def __init__(
        self,
        output_size: int,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        dropout: float = 0.2,
        vocab_size: Optional[int] = None,
        task: str = "binary",
        metrics: List[str] = ["auroc"],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()

        self.task = task
        self.dropout = dropout
        self.encoder_ckpt = encoder_ckpt
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.freeze_encoder = freeze_encoder

        # Load Encoder Model
        if Path(encoder_ckpt).exists():
            from ..utils.ckpt import get_ckpt_tokenizer

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
        if task == "binary":
            self.lossfn = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif task == "regression":
            self.lossfn = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown task type {task}")

        # Additional Metrics
        metrics = MetricCollection(
            {metric: get_metric(metric, task, output_size) for metric in metrics}
        )
        unk_token_id = load_tokenizer(encoder_ckpt).unk_token_id
        self.train_metrics = OOVMetric(metrics.clone(prefix="train/"), unk_token_id)
        self.val_metrics = OOVMetric(metrics.clone(prefix="val/"), unk_token_id)
        self.test_metrics = OOVMetric(metrics.clone(prefix="test/"), unk_token_id)

    def setup(self, stage: str) -> None:
        """Setup additional summary stats for logging"""
        for m in [self.train_metrics, self.val_metrics, self.test_metrics]:
            record_summary_stats(self.logger, m)

    def on_train_start(self):
        """Standardized training data"""
        if self.task == "regression" and self.trainer.datamodule is not None:
            ds = self.trainer.datamodule.train_dataset
            target = torch.tensor(ds.to_pandas()["target"])
            target_mean = target.mean(dim=0, keepdim=True)
            target_std = target.std(dim=0, keepdim=True)
            self.transform = Standardize(target_mean, target_std)
        else:
            self.transform = None

    def forward(self, batch, transform=True, **kwargs):  # type: ignore[override]
        hs = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            **kwargs,
        ).last_hidden_state

        pred_unscaled = self.task_network(hs)
        if self.transform and transform:
            self.transform.forward(pred_unscaled)
        return pred_unscaled

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds = self.forward(batch, transform=False)
        target = batch["target"]
        if self.transform:
            target = self.transform.inverse(target)

        loss = masked_loss(self.lossfn, preds, target, batch["target_mask"])

        if self.transform:
            preds = self.transform.forward(preds)
        return preds, loss

    def training_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        out = masked_metric_forward(
            self.train_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
            batch["input_ids"],
            batch.get("is_oov", None),
        )
        out["train/loss"] = loss
        self.log_dict(
            out,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        out = masked_metric_forward(
            self.val_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
            batch["input_ids"],
            batch.get("is_oov", None),
        )
        out["val/loss"] = loss
        self.log_dict(
            out,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        out = masked_metric_forward(
            self.test_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
            batch["input_ids"],
            batch.get("is_oov", None),
        )
        out["test/loss"] = loss
        self.log_dict(
            out,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
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
