from math import sqrt, floor
from typing import Callable, Optional, Literal
from contextlib import contextmanager

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from transformers import PreTrainedModel, AutoModel

from .model_utils import load_encoder
from ..utils.metrics import AliveFeatures, FeatureDensity, MaxFeatureDensity


def init_bias(bias, w):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


def avg_l0_norm(x: torch.Tensor) -> torch.FloatTensor:
    """Count the average number of active  features for tensor (*, F), where F is are the feature activations"""
    return x.detach().count_nonzero() / x.shape[:-1].numel()


def hf_cross_entropy(logits: torch.FloatTensor, target: torch.IntTensor):
    """F.cross_entropy but for logits of `(B, T, C)` and target of `(B, T)`"""
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


class AbstractSAE(nn.Module):
    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x into a feature representation"""
        raise NotImplementedError()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features into a reconstruction of x"""
        raise NotImplementedError()

    def forward_with_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with loss calculation -> `(self.forward(x), loss)`"""
        return self.forward(x), torch.tensor(0.0)


InjectedCoderState = Literal["null", "dense", "sparse"]


class InjectedCoder(nn.Module):
    def __init__(self, dense_model: nn.Module, coder: AbstractSAE):
        super().__init__()
        self.dense_model = dense_model
        self.coder = coder
        self.loss = torch.tensor(0.0)
        self.state: InjectedCoderState = "sparse"

    def forward(self, *args, **kwargs):
        if self.state == "null":
            return (*args,)

        output = self.dense_model(*args, **kwargs)
        if self.state == "dense":
            return output

        assert self.state == "sparse"
        hs = output[0]
        if self.training:
            hs, loss = self.coder.forward_with_loss(hs)
            self.loss = loss

        else:
            hs = self.coder.forward(hs)

        return (hs, *output[1:])


class SparsifiedModel(nn.Module):
    def __init__(self, model: nn.Module, coders: list[InjectedCoder]):
        super().__init__()
        self.model = model
        self.coders = coders

    @classmethod
    def from_huggingface(
        cls, model: PreTrainedModel, coder: AbstractSAE, layer: float | int = 0.5
    ):
        layers = model.base_model.encoder.layer
        if isinstance(layer, float):
            n_layers = len(layers)
            layer = floor(layer * n_layers)

        coders = [cls.inject_sparse_coder(layers, coder, layer)]
        return cls(model, coders)

    @staticmethod
    def inject_sparse_coder(layers: nn.ModuleList, coder: AbstractSAE, layer: int):
        m = InjectedCoder(layers[layer], coder)
        layers[layer] = m
        return m

    def set_sparsity(self, enable: bool = True):
        """Enable or disable sparse coders"""
        assert isinstance(enable, bool)
        for coder in self.coders:
            coder.state = "sparse" if enable else "dense"

    @contextmanager
    def sparse(self, enable: bool = True):
        sparsity = []
        for coder in self.coders:
            sparsity.append(coder.state)
            coder.state = "sparse" if enable else "dense"

        try:
            yield self
        finally:
            for coder in self.coders:
                coder.state = sparsity.pop()

    @contextmanager
    def nullcoders(self):
        state = []
        for coder in self.coders:
            state.append(coder.state)
            coder.state = "null"

        try:
            yield self
        finally:
            for coder in self.coders:
                coder.state = state.pop()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model.forward(input_ids, attention_mask)[0]

    def forward_with_loss(self, *args, **kwargs):
        y = self.model.forward(*args, **kwargs)[0]
        device = self.coders[0].loss.device
        loss = torch.tensor(0.0, device=device)
        for coder in self.coders:
            loss += coder.loss

        return y, loss

    def sparse_parameters(self):
        """Return an iterator over the parameters of the sparse autoencoders"""
        for coder in self.coders:
            yield from coder.coder.parameters()

    def sparse_named_parameters(self):
        for coder in self.coders:
            yield from coder.coder.named_parameters()

    @torch.no_grad()
    def loss_recovered(
        self,
        target: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lossfn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hf_cross_entropy,
        sparse_output: Optional[torch.Tensor] = None,
    ):
        """Compute the fraction of the loss recovered by the SAE relative to an null-embedding"""
        training = self.training
        self.eval()
        if sparse_output is None:
            with self.sparse(True) as self:
                sparse_output = self.forward(input_ids, attention_mask=attention_mask)
        loss_sparse = lossfn(sparse_output, target)

        with self.sparse(False) as self:
            y_dense = self.forward(input_ids, attention_mask=attention_mask)
            loss_dense = lossfn(y_dense, target)

        with self.nullcoders() as self:
            y_null = self.forward(input_ids, attention_mask=attention_mask)
            loss_null = lossfn(y_null, target)

        self.train(training)

        return 1 - (loss_sparse - loss_dense) / (loss_null - loss_dense)


class GatedSAE(AbstractSAE):
    def __init__(
        self,
        hidden_size: int,
        expansion: int = 4,
        l1_coef: float = 0.01,
        device=None,
        dtype=None,
    ):
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
        self.l1_coef = torch.tensor(l1_coef, requires_grad=False, **factory_kwargs)

        # Loss parameters
        self.loss = torch.tensor(0.0, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_gate, self.w_dec]:
            nn.init.kaiming_uniform_(w, nonlinearity="relu")

        init_bias(self.b_gate, self.w_gate)
        init_bias(self.b_dec, self.w_dec)
        # kaiming init for r_mag
        nn.init.normal_(self.r_mag, mean=0.0, std=sqrt(2 / self.r_mag.shape[0]))
        init_bias(self.b_enc, self.w_gate)

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    def encode(self, x: torch.Tensor):
        x_centered = x - self.b_dec
        x_enc = x_centered.matmul(self.w_gate.T)
        gate = (x_enc + self.b_gate) > 0
        x_mag = F.relu(self.r_mag.exp() * x_enc + self.b_enc)
        return x_mag * gate

    def decode(self, features: torch.Tensor):
        return F.linear(features, self.w_dec, self.b_dec)

    def forward_with_loss(self, x: torch.Tensor):
        x_centered = x - self.b_dec
        x_enc = x_centered.matmul(self.w_gate.T)
        pi_gate = x_enc + self.b_gate
        pi_rect = F.relu(pi_gate)
        loss_sparsity = self.l1_coef * pi_rect.abs().sum()

        x_mag = F.relu(self.r_mag.exp() * x_enc + self.b_enc)
        gate = pi_gate > 0
        features = x_mag * gate
        x_hat = F.linear(features, self.w_dec, self.b_dec)
        loss_recon = F.mse_loss(x_hat, x)

        x_hat_detach = F.linear(pi_rect, self.w_dec.detach(), self.b_dec.detach())
        loss_aux = F.mse_loss(x_hat_detach, x)

        loss = loss_sparsity + loss_aux + loss_recon
        return x_hat, loss


class TiedBiasSAE(AbstractSAE):
    def __init__(self, hidden_size: int, expansion: int = 4, l1_coef: float = 0.01):
        super().__init__()
        n_features = hidden_size * expansion
        self.encoder = nn.Linear(hidden_size, n_features)
        self.decoder = nn.Linear(n_features, hidden_size)
        self.l1_coef_coef = torch.tensor(l1_coef)

    def encode(self, x: torch.Tensor):
        return self.encoder(x - self.decoder.bias)

    def decode(self, features):
        return self.decoder(features)

    def forward_with_loss(self, x: torch.Tensor):
        f = F.relu(self.encoder(x - self.decoder.bias))
        x_hat = self.decoder(f)
        f_act = f * self.decoder.weight.norm(p=2, dim=0)
        loss_reconstruction = F.mse_loss(x_hat, x)
        loss_sparsity = self.l1_coef_coef * f_act.abs().sum()
        loss = loss_reconstruction + loss_sparsity
        return x_hat, loss


def pretrained_sae(
    name_or_path: str, sae: str = "gated", layer: int | float = 0.5, expansion: int = 2
):
    model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
    hidden_size = model.config.hidden_size
    if sae == "gated":
        sae = GatedSAE(hidden_size=hidden_size, expansion=expansion)


class LightningSAE(pl.LightningModule):
    def __init__(
        self,
        name_or_path: str,
        sae_type: str = "gated",
        layer: int | float = 0.5,
        l1_coef: float = 0.01,
        expansion: int = 2,
        lossfn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hf_cross_entropy,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()

        encoder = load_encoder(name_or_path)
        hidden_size = encoder.config.hidden_size
        if sae_type == "gated":
            sae = GatedSAE(
                hidden_size=hidden_size, expansion=expansion, l1_coef=l1_coef
            )
        else:
            raise ValueError(f"Unknown sae type: {sae_type}")

        self.sparse_model = SparsifiedModel.from_huggingface(encoder, sae, layer)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lossfn = lossfn
        self.save_hyperparameters(ignore=["sparse_model"])

        # metrics = MetricCollection(
        #     {
        #         "alive_features": AliveFeatures(self.num_features),
        #         "alive_features_1k": AliveFeatures(self.num_features, threshold=1_000),
        #         "alive_features_10k": AliveFeatures(
        #             self.num_features, threshold=10_000
        #         ),
        #         "max_feature_density": MaxFeatureDensity(self.num_features),
        #         "feature_density": FeatureDensity(self.num_features),
        #     }
        # )
        # self.train_metrics = metrics.clone(prefix="train/")
        # self.val_metrics = metrics.clone(prefix="val/")
        # self.test_metrics = metrics.clone(prefix="test/")

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

    def stage_step(self, stage: str, batch, recovered_loss: bool = False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y, loss = self.sparse_model.forward_with_loss(input_ids, attention_mask)
        logdict = {f"{stage}/loss": loss}

        if self.lossfn:
            target = batch["target"] if "target" in batch else batch["labels"]
            y_loss = self.lossfn(y, target)
            logdict[f"{stage}/target_loss"] = y_loss

            if recovered_loss:
                logdict[f"{stage}/recovered_loss"] = self.sparse_model.loss_recovered(
                    target,
                    input_ids,
                    attention_mask,
                    lossfn=self.lossfn,
                    sparse_output=y,
                )

        return loss, logdict

    def training_step(self, batch):
        loss, logdict = self.stage_step("train", batch)
        self.log_dict(logdict, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch):
        loss, logdict = self.stage_step("val", batch, recovered_loss=True)
        logdict = {k + "_epoch": v for k, v in logdict.items()}
        self.log_dict(logdict, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch):
        loss, logdict = self.stage_step("test", batch, recovered_loss=True)
        logdict = {k + "_epoch": v for k, v in logdict.items()}
        self.log_dict(logdict, on_epoch=True, on_step=False)
        return loss

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
                data=list(zip(feature_density.bin_centers, feature_density.density)),
                columns=["bin_center", "density"],
            )

        metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.sparse_model.sparse_parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
