from itertools import chain
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torchmetrics import MetricCollection
from sklearn.preprocessing import PowerTransformer as _PowerTransformer

from ..utils.metrics import (
    OOVMetric,
    get_metric,
    masked_loss,
    masked_metric_update,
)
from .model_utils import record_summary_stats
from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin
from .prediction_task_head import PredictionTaskHead


class Standardize(torch.nn.Module):
    def __init__(self, num_outputs: int, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.std * x) + self.mean

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def fit(self, ds) -> dict:
        target = torch.stack([torch.tensor(x) for x in ds["target"]])
        print(f"target: {target.shape}")
        self.mean = target.mean(0).to(self.mean)
        self.std = target.std(0).to(self.std) + self.eps
        return self.state_dict()


class PowerTransform(torch.nn.Module):
    """
    Apply a power transform (Yeo-Johnson) featurewise to make data more Gaussian-like.
    Followed by applying a zero-mean, unit-variance normalization to the
    transformed output to rescale targets to [-1, 1].
    """

    def __init__(self, num_outputs, eps: float = 1e-8):
        super().__init__()
        self.num_outputs = num_outputs
        self.register_buffer("lmbdas", torch.zeros(num_outputs))
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def _yeo_johnson_transform(self, x, lmbda):
        """
        Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3354
        """
        x_out = x.clone()
        eps = torch.finfo(x.dtype).eps
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < eps:
            x_out[pos] = torch.log1p(x[pos])
        else:  # lmbda != 0
            x_out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > eps:
            x_out[~pos] = -(torch.pow(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            x_out[~pos] = -torch.log1p(-x[~pos])

        return x_out

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """
        Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3383
        """
        x_out = x.clone()
        pos = x >= 0
        eps = torch.finfo(x.dtype).eps

        # when x >= 0
        if abs(lmbda) < eps:  # lmbda == 0
            x_out[pos] = torch.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_out[pos] = torch.pow(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > eps:  # lmbda != 2
            x_out[~pos] = 1 - torch.pow(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_out[~pos] = 1 - torch.exp(-x[~pos])
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo standardization
        x = (self.std * x) + self.mean
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_inverse_transform(x[:, i], self.lmbdas[i])
        return x_out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_transform(x[:, i], self.lmbdas[i])
        # Standardization
        x_out = (x_out - self.mean) / self.std
        return x_out

    def fit(self, ds) -> dict:
        target = torch.stack([torch.tensor(x) for x in ds["target"]])
        # Fit Yeo-Johnson lambdas
        transformer = _PowerTransformer(method="yeo-johnson", standardize=False)
        target = torch.tensor(transformer.fit_transform(target))
        self.lmbdas = torch.tensor(transformer.lambdas_)
        # Fit standardization scaling
        self.mean = target.mean(0).to(self.mean)
        self.std = target.std(0).to(self.std) + self.eps
        return self.state_dict()


class IdentityTransform(torch.nn.Identity):
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
        transform: Optional[str] = None,
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
            transform = transform or "standardize"
        else:
            raise ValueError(f"Unknown task type {task}")

        # Init Transform
        if transform == "standardize":
            self.transform = Standardize(output_size)
        elif transform == "power_transform":
            self.transform = PowerTransform(output_size)
        else:
            self.transform = IdentityTransform()
        self.transform.eval()

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
            preds,
            batch["target"],
            batch["target_mask"],
            batch["input_ids"],
            batch.get("is_oov", None),
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

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
