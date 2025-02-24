from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from ..utils.metrics import (
    OOVMetric,
    bootstrap_collection,
    get_metrics,
    masked_loss,
    masked_metric_update,
)
from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin, record_loss_summary_stats, record_summary_stats
from .normalize import get_normalizer
from .prediction_task_head import PredictionTaskHead


def load_encoder(encoder: str | Path | torch.nn.Module):
    if isinstance(encoder, torch.nn.Module):
        return encoder
    elif (
        Path(encoder).exists()
        and Path(encoder).parent.parent.joinpath("config.json").is_file()
    ):
        return DeepSpeedMixin.load(encoder).get_encoder()
    else:
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            encoder, trust_remote_code=True, add_pooling_layer=False
        )


class LMFinetuning(LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on multiple tasks.
    """

    def __init__(
        self,
        output_size: int,
        encoder_ckpt: str,
        freeze_encoder: bool | str = False,
        dropout: float = 0.2,
        vocab_size: Optional[int] = None,
        task: str = "binary",
        metrics: List[str] = ["auroc"],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str] = None,
        tokenizer: Optional[str] = None,
        bootstrap: Union[bool, int] = False,
        target_columns: Optional[List[str]] = None,
        track_oov: bool = True,
    ) -> None:
        super().__init__()

        self.task = task
        self.output_size = output_size
        self.dropout = dropout
        self.encoder_ckpt = encoder_ckpt
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.freeze_encoder = freeze_encoder

        if self.task == "binary":
            self.lossfn = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif self.task == "regression":
            self.lossfn = torch.nn.MSELoss(reduction="none")
            transform = transform or "standardize"
        else:
            raise ValueError(f"Unknown task type {self.task}")
        self.transform = get_normalizer(transform, self.output_size).eval()

        self.save_hyperparameters()

        # Additional Metrics
        metrics = get_metrics(
            metrics,
            task,
            num_outputs=output_size,
            target_channels=target_columns,
        )

        if bootstrap:
            n = bootstrap if isinstance(bootstrap, int) else 50
            metrics = bootstrap_collection(metrics, num_bootstraps=n)

        if track_oov:
            unk_token_id = load_tokenizer(tokenizer or encoder_ckpt).unk_token_id
            self.train_metrics = OOVMetric(metrics.clone(prefix="train/"), unk_token_id)
            self.val_metrics = OOVMetric(metrics.clone(prefix="val/"), unk_token_id)
            self.test_metrics = OOVMetric(metrics.clone(prefix="test/"), unk_token_id)
        else:
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

    def configure_model(self):
        self.encoder = load_encoder(self.encoder_ckpt)
        self.task_network = PredictionTaskHead(
            embed_dim=self.encoder.config.hidden_size,
            output_size=self.output_size,
            dropout=self.dropout,
        )

    def setup(self, stage: str) -> None:
        """Setup additional summary stats for logging"""
        for m in [self.train_metrics, self.val_metrics, self.test_metrics]:
            record_summary_stats(self.logger, m)
        record_loss_summary_stats(self.logger)

    def on_fit_start(self):
        """Standardized training data"""
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
        target = self.transform.inverse(target)
        loss = masked_loss(self.lossfn, preds, target, batch["target_mask"])
        preds = self.transform.forward(preds)
        return preds, loss

    def training_step(self, batch, batch_idx: int):
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        masked_metric_update(
            self.train_metrics,
            preds,
            batch["target"],
            batch["target_mask"],
            batch["input_ids"],
            batch.get("is_oov", None),
            int_cast=self.task == "binary",
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            self.train_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx: int):
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
            batch["input_ids"],
            batch.get("is_oov", None),
            int_cast=self.task == "binary",
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int):
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
            batch["input_ids"],
            batch.get("is_oov", None),
            int_cast=self.task == "binary",
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
        hs = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).last_hidden_state
        embedding = hs[:, 0, :]

        preds = self.task_network(hs)
        preds = self.transform.forward(preds)

        out = {"embedding": embedding, "prediction": preds}
        for key in ["target", "is_oov"]:
            if key in batch.keys():
                out[key] = batch[key]

        return out

    def configure_optimizers(self):
        learnable_params = self.task_network.parameters()
        if not self.freeze_encoder:
            learnable_params = chain(learnable_params, self.encoder.parameters())
        elif self.freeze_encoder == "encoder":
            learnable_params = chain(
                learnable_params, self.encoder.embeddings.parameters()
            )

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
