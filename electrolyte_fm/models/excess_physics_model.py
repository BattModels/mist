from itertools import chain

import torch
from torch import nn
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger

from .model_utils import masked_mean_pool
from ..utils.metrics import (
    get_metrics,
    masked_loss,
    masked_metric_update,
)
from .normalize import AbstractNormalizer, IdentityTransform
from .polynomial_task_head import LagrangePolynomial


def pairwise_fusion(name: str, *args, **kwargs):
    if name == "square-difference":
        return PairwiseInteraction(*args, **kwargs)
    elif name == "gaussian":
        return GaussianFusion(*args, **kwargs)
    elif name == "difference":
        return EquivariantInteraction(*args, **kwargs)
    elif name == "softmax":
        return SoftmaxFusion(*args, **kwargs)
    else:
        raise ValueError(f"Unknown fusion: {name}")


class PairwiseInteraction(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_targets: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_targets = n_targets
        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(n_in, n_out * n_targets),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        y = self.mlp(self.distance(a, b))
        y = y.reshape(*y.shape[:-1], self.n_targets, self.n_out)
        return y + y.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return (a - b).pow(2)


class GaussianFusion(PairwiseInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = -1 * (a - b).pow(2)
        return dist.exp()


class EquivariantInteraction(PairwiseInteraction):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        emb_a, emb_b = self.distance(a, b)
        y_a = self.mlp(emb_a).reshape(*emb_a.shape[:-1], self.n_targets, self.n_out)
        y_b = self.mlp(emb_b).reshape(*emb_a.shape[:-1], self.n_targets, self.n_out)
        return y_a + y_b.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return a - b, b - a


class SoftmaxFusion(EquivariantInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor):
        y = torch.stack([a, b]).log_softmax(dim=0)
        return y[0], y[1]


class ExcessPhysicsModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        pairwise_interaction: PairwiseInteraction,
        component_properties: nn.Module,
        temperature_dependence: list | None = None,
        relative_excess: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pairwise_interaction = pairwise_interaction
        self.component_properties = component_properties
        self.excess_polynomial = LagrangePolynomial(
            polynomial_order=self.pairwise_interaction.n_out + 2,
            zero_endpoints=True,
        )
        self.temperature_dependence = torch.tensor(temperature_dependence or 1.0)
        self.relative_excess = relative_excess

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        composition: torch.Tensor,
        temperature: torch.Tensor,
        transform: bool = True,
    ):
        # Get component embedding vectors
        hs = self.encoder(
            input_ids.reshape(-1, input_ids.shape[-1]),
            attention_mask=attention_mask.reshape(-1, attention_mask.shape[-1]),
            return_dict=True,
        ).last_hidden_state.reshape(*input_ids.shape, -1)

        # Mean pool over tokens (B, C, L, E) -> (B, C, E)
        embs = masked_mean_pool(hs, attention_mask)

        # Compute Pairwise interactions
        B, C, E = embs.shape
        indices = torch.triu_indices(C, C, offset=1)
        I = indices.shape[1]  # noqa: E741
        e_i = embs[:, indices[0]].reshape(B * I, E)
        e_j = embs[:, indices[1]].reshape(B * I, E)
        pw_coeffs = self.pairwise_interaction(e_i, e_j)  # (B*I, E) -> (B*I, T, P)
        T = self.pairwise_interaction.n_targets
        P = self.pairwise_interaction.n_out
        assert pw_coeffs.shape == (B * I, T, P)

        # Compute partial pairwise concentrations
        # I.e. Partial concentration of i assuming an i+j mixture
        x_t = composition[:, indices[0]] + composition[:, indices[1]]
        x_i = composition[:, indices[0]] / x_t
        x_i = x_i.reshape(B * I, 1)

        # Evaluate interaction polynomials at compositions
        pw = self.excess_polynomial(pw_coeffs, x_i)
        assert pw.shape == (B * I, T)

        # Sum over interactions to get activity coefficients
        gamma = pw.reshape(B, I, -1).sum(dim=1)
        assert gamma.shape == (B, T)

        # Inject Temperature dependence
        gamma = gamma * temperature.view(B, 1).pow(self.temperature_dependence)

        # Predict Pure Property (B, C, E) -> (B, C, T)
        y_target = self.component_properties(embs)
        assert y_target.shape == (B, C, T)

        # Linear Mixing (B, C, T) -> (B, T)
        assert composition.shape == (B, C)
        y_linear = (y_target * composition.view(B, C, 1)).sum(dim=1)

        if self.relative_excess:
            y = y_linear * gamma.exp()
            y_excess = y - y_linear
        else:
            y_excess = gamma
            y = y_linear + gamma

        return y, y_linear, y_excess


class ExcessPhysicsLightningModel(LightningModule):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        freeze_encoder: bool = False,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: AbstractNormalizer | None = None,
        metrics: list[str] = ["mae", "rmse", "mape"],
        target_columns: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.freeze_encoder = freeze_encoder
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.transform = transform or IdentityTransform()

        metrics = get_metrics(
            metrics,
            "regression",
            target_columns=target_columns,
            num_outputs=len(target_columns) if target_columns else 1,
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min,last")

    def on_fit_start(self):
        """Standardized training data"""
        state = None
        if self.global_rank == 0:
            assert self.trainer.datamodule.target_dataset is not None
            ds = self.trainer.datamodule.target_dataset
            state = self.transform.fit(ds)

        state = self.trainer.strategy.broadcast(state)
        self.transform.load_state_dict(state)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_loss(self, **kwargs):
        y, _, y_excess = self.forward(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            temperature=kwargs["temperature"],
            composition=kwargs["composition"],
        )

        # Mixture Property Prediction
        loss_mix = masked_loss(self.lossfn, y, kwargs["target"], kwargs["target_mask"])
        if "target_excess" in kwargs:
            loss_excess = masked_loss(
                self.lossfn,
                y_excess,
                kwargs["target_excess"],
                kwargs["target_excess_mask"],
            )
        else:
            loss_excess = torch.tensor(0.0)
        loss = loss_mix + loss_excess

        return y, loss

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
