from math import sqrt
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection

from ..utils.ckpt import get_hidden_size
from ..utils.metrics import AliveFeatures, FeatureDensity, MaxFeatureDensity
from .model_utils import load_encoder


def init_bias(bias, w):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


def avg_l0_norm(x: torch.Tensor) -> torch.FloatTensor:
    """Count the average number of active  features for tensor (*, F), where F is are the feature activations"""
    return x.detach().count_nonzero() / x.shape[:-1].numel()


class GatedSAE(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        n_features = hidden_size * expansion
        self.w_gate = nn.Parameter(
            torch.empty((n_features, hidden_size), **factory_kwargs)
        )
        self.b_gate = nn.Parameter(torch.empty(n_features, **factory_kwargs))
        self.w_dec = nn.Parameter(
            torch.empty((hidden_size, n_features), **factory_kwargs)
        )
        self.b_dec = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.r_mag = nn.Parameter(torch.empty(n_features, **factory_kwargs))
        self.b_enc = nn.Parameter(torch.empty(n_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_gate, self.w_dec]:
            nn.init.kaiming_uniform_(w, nonlinearity="relu")

        init_bias(self.b_gate, self.w_gate)
        init_bias(self.b_dec, self.w_dec)
        # kaiming init for r_mag
        nn.init.normal_(self.r_mag, mean=0.0, std=sqrt(2 / self.r_mag.shape[0]))
        init_bias(self.b_enc, self.w_gate)

    def forward(self, x):
        x_centered = x - self.b_dec
        x_enc = x_centered.matmul(self.w_gate.T)
        gate = (x_enc + self.b_gate) > 0
        x_mag = F.relu(self.r_mag.exp() * x_enc + self.b_enc)
        return x_mag * gate

    def reconstruct(self, features):
        return F.linear(features, self.w_dec, self.b_dec)

    @torch.compile
    def loss(self, x, l1_coef: torch.FloatTensor = 0.01):
        x_centered = x - self.b_dec
        x_enc = x_centered.matmul(self.w_gate.T)
        pi_gate = x_enc + self.b_gate
        pi_rect = F.relu(pi_gate)
        loss_sparsity = l1_coef * pi_rect.abs().sum()

        x_mag = F.relu(self.r_mag.exp() * x_enc + self.b_enc)
        gate = pi_gate > 0
        features = x_mag * gate
        x_hat = F.linear(features, self.w_dec, self.b_dec)
        loss_recon = F.mse_loss(x_hat, x)

        x_hat_detach = F.linear(pi_rect, self.w_dec.detach(), self.b_dec.detach())
        loss_aux = F.mse_loss(x_hat_detach, x)

        return {
            "loss": loss_sparsity + loss_aux + loss_recon,
            "features": features,
        }


class TiedBiasSAE(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        n_features = hidden_size * expansion
        self.encoder = nn.Linear(hidden_size, n_features)
        self.decoder = nn.Linear(n_features, hidden_size)

    def forward(self, x):
        return F.relu(self.encoder(x - self.decoder.bias))

    def reconstruct(self, features):
        return self.decoder(features)

    @torch.compile
    def loss(self, x, l1_coef: torch.FloatTensor = 0.01):
        f = F.relu(self.encoder(x - self.decoder.bias))
        x_hat = self.decoder(f)
        f_act = f * self.decoder.weight.norm(p=2, dim=0)
        loss = F.mse_loss(x_hat, x) + l1_coef * f_act.abs().sum()
        return {"loss": loss, "features": f}


class SAE(pl.LightningModule):
    def __init__(
        self,
        hidden_size: Union[int, str],
        sae: str = "gated",
        expansion: int = 4,
        l1_coef: float = 0.01,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()
        if isinstance(hidden_size, str):
            hidden_size = get_hidden_size(hidden_size)

        if sae == "gated":
            self.sae = GatedSAE(hidden_size, expansion)
        elif sae == "tied_bias":
            self.sae = TiedBiasSAE(hidden_size, expansion)
        else:
            raise ValueError(f"Unknown sae type: {sae}")

        self.l1_coef = l1_coef
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.hidden_size = hidden_size
        self.num_features = hidden_size * expansion
        self.save_hyperparameters(ignore=["sae"])

        metrics = MetricCollection(
            {
                "alive_features": AliveFeatures(self.num_features),
                "max_feature_density": MaxFeatureDensity(self.num_features),
                "feature_density": FeatureDensity(self.num_features),
            }
        )
        print(metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def setup(self, stage: str):
        if isinstance(self.logger, pl.loggers.WandbLogger):

            def define_metric(*args, **kwargs):
                self.logger.experiment.define_metric(*args, **kwargs)

            # Add summary metrics for loss
            for stage in ["train", "val", "test"]:
                for s in ["step", "epoch"]:
                    define_metric(
                        f"{stage}/loss_{s}", summary="best,min,last", goal="minimize"
                    )

                define_metric(f"{stage}/alive_features", summary="max,last")
                define_metric(f"{stage}/max_feature_density", summary="min,max,last")

    def stage_step(self, stage: str, batch):
        out = self.sae.loss(batch, self.l1_coef)
        self.log_dict(
            {
                f"{stage}/loss": out["loss"],
                f"{stage}/avg_l0_loss": avg_l0_norm(out["features"]),
            },
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        return out

    def backward(self, loss, *args, **kwargs) -> None:
        # Retain graph during backprop
        loss.backward(retain_graph=True)

    def training_step(self, batch):
        out = self.stage_step("train", batch)
        self.train_metrics.update(out["features"])
        return out["loss"]

    def validation_step(self, batch):
        out = self.stage_step("val", batch)
        self.val_metrics.update(out["features"])
        return out["loss"]

    def test_step(self, batch):
        out = self.stage_step("test", batch)
        self.test_metrics.update(out["features"])
        return out["loss"]

    def _log_feature_metrics(self, metric, stage: str) -> None:
        m = metric.compute()
        feature_density = m.pop(stage + "/feature_density", None)
        self.log_dict(m, on_epoch=True, sync_dist=True)
        if (
            isinstance(self.logger, pl.loggers.WandbLogger)
            and feature_density is not None
        ):
            self.logger.log_table(
                stage + "/feature_density",
                data=list(
                    zip(feature_density.bin_centers, feature_density.density)
                ),
                columns=["bin_center", "density"],
            )

        metric.reset()

    def on_train_epoch_end(self):
        self._log_feature_metrics(self.train_metrics, "train")

    def on_validation_epoch_end(self):
        self._log_feature_metrics(self.val_metrics, "val")

    def on_test_epoch_end(self):
        self._log_feature_metrics(self.test_metrics, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
