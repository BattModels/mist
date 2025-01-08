import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanAbsoluteError
import math
from .model_utils import CanSkip, DeepSpeedMixin, LoggingMixin
from ..utils.metrics import TokenCounter
from itertools import chain
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torchmetrics import Metric

from ..utils.metrics import (
    OOVMetric,
    get_metric,
    masked_loss,
    masked_metric_update,
)
from .model_utils import record_summary_stats
from ..utils.tokenizer import load_tokenizer


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
        self.mean = target.float().mean(0).to(self.mean)
        self.std = target.float().std(0).to(self.std) + self.eps
        return self.state_dict()


class RKPredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.n_components = n_components
        embed_dim += 1  # Temperature appended to embedding

        # predict property for single substance in mixture (P_i)
        self.single_substance_property = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # predict R-K polynomial coefficients for pairs of molecules
        # (C^k_ij)
        self.RK_coeffients = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, polynomial_order),
        )

    def forward(self, batch):
        P_m = 0

        # linear mixing term
        for i in range(self.n_components):
            P_i = self.single_substance_property(batch[f"embedding_{i}"])
            P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        concat_embedding = tuple(
            batch[f"embedding_{i}"] for i in range(self.n_components)
        )

        concat_embedding = torch.hstack(concat_embedding)

        RK_coeffients = self.RK_coeffients(concat_embedding)

        # excess term
        for i in range(self.n_components):
            x_i = batch[f"composition_{i}"]
            for j in range(i + 1, self.n_components):
                x_j = batch[f"composition_{j}"]
            for k in range(self.polynomial_order):
                x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]
                RK_summation = torch.mul(
                    torch.mul(-(1.0**k), RK_coeffients[:, k]), torch.pow((x_i - x_j), k)
                )
                P_m += torch.mul(x_ix_j, RK_summation).view(-1, 1)

        return P_m  # [batch_size, 1]


class LegendrePredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.n_components = n_components
        embed_dim += 1  # Temperature appended to embedding

        # predict property for single substance in mixture (P_i)
        self.single_substance_property = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.coeffients = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, polynomial_order),
        )

    def forward(self, batch):
        P_m = 0

        # linear mixing term
        for i in range(self.n_components):
            P_i = self.single_substance_property(batch[f"embedding_{i}"])
            P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        concat_embedding = tuple(
            batch[f"embedding_{i}"] for i in range(self.n_components)
        )

        concat_embedding = torch.hstack(concat_embedding)

        coeffients = self.coeffients(concat_embedding)

        # binary excess term
        x_i = batch["composition_0"]
        x_j = batch["composition_1"]
        x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]

        x = 2 * x_j - 1
        M = math.floor(0.5 * self.polynomial_order)

        for m in range(M):
            summation = torch.mul(
                torch.mul(-(1.0**m), coeffients[:, m]),
                torch.pow(x, self.polynomial_order - 2 * m),
            )

            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # [batch_size, 1]


class ChebyshevPredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.n_components = n_components
        embed_dim += 1  # Temperature appended to embedding

        # predict property for single substance in mixture (P_i)
        self.single_substance_property = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # predict polynomial coefficients for pairs of molecules
        self.coeffients = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, self.n_components * embed_dim),
            nn.ReLU(),
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, polynomial_order),
        )

    def forward(self, batch):
        P_m = 0

        # linear mixing term
        for i in range(self.n_components):
            P_i = self.single_substance_property(batch[f"embedding_{i}"])
            P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        concat_embedding = tuple(
            batch[f"embedding_{i}"] for i in range(self.n_components)
        )

        concat_embedding = torch.hstack(concat_embedding)

        coeffients = self.coeffients(concat_embedding)

        # binary excess term
        x_i = batch["composition_0"]
        x_j = batch["composition_1"]
        x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]

        x = 2 * x_j - 1
        M = math.floor(0.5 * self.polynomial_order)

        def chebyshev_poly(n, x):
            """Recursive formula to compute Chebyshev Polynomials T_n(x)"""
            if n == 0:
                return torch.ones_like(x)
            elif n == 1:
                return x
            else:
                return torch.mul(2 * x, chebyshev_poly(n - 1, x)) - chebyshev_poly(
                    n - 2, x
                )

        for m in range(M):
            summation = torch.mul(
                torch.mul((-1.0) ** m, coeffients[:, m]),
                chebyshev_poly(self.polynomial_order - 2 * m, x),
            )

            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # [batch_size, 1]


def metric_update(
    metrics: Metric,
    preds: torch.Tensor,
    targets: torch.Tensor,
    *args,
):
    """Update metrics"""
    metrics.update(preds, targets, *args)


class MixtureModelWithPhysics(LightningModule, DeepSpeedMixin, LoggingMixin):
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
        temperature = batch["temperature"]
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)
            embedding = torch.hstack((temperature.view(-1, 1), embedding))
            batch[f"embedding_{i}"] = embedding
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
