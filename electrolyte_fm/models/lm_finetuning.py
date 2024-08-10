from itertools import chain
from pathlib import Path
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable

from .model_utils import DeepSpeedMixin
from .prediction_task_head import PredictionTaskHead
from ..utils.metrics import get_metric, masked_loss, masked_metric_forward


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
        task: str = "binary_classification",
        metrics: list[str] = ["auroc"],
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

        self.save_hyperparameters()

        self.task_network = PredictionTaskHead(
            embed_dim=self.encoder.config.hidden_size,
            output_size=output_size,
            dropout=dropout,
        )
        if task == "binary_classification":
            self.lossfn = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif task == "regression":
            self.lossfn = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown task type {task}")

        # Additional Metrics
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = get_metric(metric, task, output_size)

    def forward(self, batch, **kwargs):  # type: ignore[override]
        hs = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            **kwargs,
        ).last_hidden_state

        return self.task_network(hs)

    def _phase_step(self, batch, batch_idx: int, phase: str) -> torch.FloatTensor:
        preds = self(batch)
        loss = self.masked_loss(preds, batch["target"], batch["target_mask"])
        out = {phase + "loss": loss}
        metrics = masked_metric_forward(
            self.metrics, preds, batch["target"], batch["target_mask"]
        )
        out.update({phase + k: v for k, v in metrics.items()})
        self.log_dict(
            out,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self._phase_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self._phase_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self._phase_step(batch, batch_idx, "test")

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
