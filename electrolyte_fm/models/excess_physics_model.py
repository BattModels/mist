import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from transformers import AutoModel, AutoConfig, PretrainedConfig
from transformers import CONFIG_MAPPING as HF_CONFIG_MAPPING

from ..utils.metrics import (
    get_metrics,
    masked_loss,
    masked_metric_update,
)
from .model_utils import masked_mean_pool
from .normalize import Standardize
from .polynomials import LagrangePolynomial


def _default_targets():
    return [
        "density [gram / centimeter ** 3]",
        "molar volume [centimeter ** 3 / mole]",
        "molar enthalpy [joule / mole]",
    ]


@dataclass
class ExcessPhysicsConfig:
    encoder: PretrainedConfig
    target_columns: list[str] | None = field(default_factory=_default_targets)
    interactions: str = "difference"
    num_control: int = 3
    temperature_dependence: list | float = 1.0
    dropout: float = 0.1

    @property
    def num_targets(self):
        return len(self.target_columns) if self.target_columns is not None else 1

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "encoder"}
        d["encoder"] = self.encoder.to_diff_dict()
        d["encoder"]["model_type"] = self.encoder.model_type
        if hasattr(self.encoder, "_name_or_path") and self.encoder._name_or_path:
            d["encoder"]["_name_or_path"] = self.encoder._name_or_path
        return d

    @classmethod
    def from_dict(cls, d: dict):
        encoder_config = d.pop("encoder")
        if (
            encoder_config["model_type"] not in HF_CONFIG_MAPPING
            and "_name_or_path" in encoder_config
        ):
            model_id = encoder_config.pop("_name_or_path")
            encoder = AutoConfig.from_pretrained(
                model_id, **encoder_config, trust_remote_code=True
            )
        else:
            encoder = AutoConfig.for_model(**encoder_config)
        return ExcessPhysicsConfig(encoder, **d)

    def to_json_file(self, config_file: str | Path):
        Path(config_file).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json_file(cls, config_file: str | Path):
        return cls.from_dict(json.loads(Path(config_file).read_text()))


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
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Reset weights of last
        last = self.mlp[-1]
        nn.init.xavier_normal_(last.weight, gain=nn.init.calculate_gain("linear"))

        self.mlp.apply(init_weights)

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
    def __init__(self, config: ExcessPhysicsConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_config(config.encoder)
        self.pairwise_interaction = pairwise_fusion(
            config.interactions,
            config.encoder.hidden_size,
            config.num_control,
            n_targets=config.num_targets,
        )
        self.component_properties = nn.Sequential(
            nn.Linear(config.encoder.hidden_size, config.encoder.hidden_size),
            nn.Dropout(config.dropout),
            nn.SiLU(),
            nn.Linear(config.encoder.hidden_size, config.num_targets),
        )
        self.excess_polynomial = LagrangePolynomial(
            polynomial_order=self.config.num_control + 2,
            zero_endpoints=True,
        )
        self.transform = Standardize(num_outputs=config.num_targets)
        self.register_buffer(
            "temperature_dependence",
            torch.tensor(
                1.0
                if config.temperature_dependence is None
                else config.temperature_dependence
            ),
        )

    def save_pretrained(self, save_directory: str | Path):
        from safetensors.torch import save_model

        save_directory = Path(save_directory)
        save_directory.mkdir(exist_ok=True, parents=True)
        save_model(self, str(save_directory.joinpath("model.safetensors")))
        self.config.to_json_file(save_directory.joinpath("config.json"))

    @classmethod
    def from_pretrained(cls, save_directory: str | Path):
        from safetensors.torch import load_model

        save_directory = Path(save_directory)
        model = cls(
            ExcessPhysicsConfig.from_json_file(save_directory.joinpath("config.json"))
        )
        load_model(model, save_directory.joinpath("model.safetensors"))
        return model

    @classmethod
    def from_pretrained_encoder(
        cls,
        name_or_path,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        encoder = AutoModel.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code
        )
        config = ExcessPhysicsConfig(encoder=encoder.config, **kwargs)
        model = cls(config)
        model.encoder = encoder
        return model

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
            output_attentions=False,
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
        x_i = torch.where(x_i.abs() > 1e-8, x_i, torch.tensor(0.0).to(x_i))
        x_i = x_i.reshape(B * I, 1)

        # Evaluate interaction polynomials at compositions
        pw = self.excess_polynomial(pw_coeffs, x_i)
        assert pw.shape == (B * I, T)

        # Sum over interactions to get excess properties
        y_excess = pw.reshape(B, I, -1).sum(dim=1)
        assert y_excess.shape == (B, T)

        # Inject Temperature dependence
        y_excess = y_excess * (temperature / 273.15).view(B, 1).pow(
            self.temperature_dependence
        )

        # Predict Pure Property (B, C, E) -> (B, C, T)
        y_target = self.component_properties(embs)
        assert y_target.shape == (B, C, T), f"Got: {y_target.shape}, vs {(B, C, T)}"

        # Linear Mixing (B, C, T) -> (B, T)
        assert composition.shape == (B, C)
        y_linear = (y_target * composition.view(B, C, 1)).sum(dim=1)

        # Transform to real-units
        if transform:
            y_linear = self.transform.forward(y_linear)
            y_excess = y_excess * self.transform.std

        y = y_linear + y_excess

        return y, y_linear, y_excess


class ExcessPhysicsLightningModel(LightningModule):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        model: ExcessPhysicsModel | ExcessPhysicsConfig,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        metrics: list[str] = ["rmse-channel", "mae-channel", "r2-channel"],
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # Load Model
        if isinstance(model, dict):
            self.model = ExcessPhysicsModel(ExcessPhysicsConfig.from_dict(model))
        elif isinstance(model, ExcessPhysicsConfig):
            self.model = ExcessPhysicsModel(model)
        elif isinstance(model, ExcessPhysicsModel):
            self.model = model
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        self.save_hyperparameters({"model": self.model.config.to_dict()})

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lossfn = nn.MSELoss(reduction="none")

        metrics = get_metrics(
            metrics,
            "regression",
            target_channels=self.model.config.target_columns,
            num_outputs=self.model.config.num_targets,
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
            state = self.model.transform.fit(ds)

        state = self.trainer.strategy.broadcast(state)
        self.model.transform.load_state_dict(state)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_loss(self, **kwargs):
        y, y_linear, y_excess = self.forward(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            temperature=kwargs["temperature"],
            composition=kwargs["composition"],
            transform=False,
        )

        # Mixture Property Prediction
        target = self.model.transform.inverse(kwargs["target"])
        loss_mix = masked_loss(self.lossfn, y, target, kwargs["target_mask"])
        if "target_excess" in kwargs:
            target_excess = kwargs["target_excess"] / self.model.transform.std
            loss_excess = masked_loss(
                self.lossfn,
                y_excess,
                target_excess,
                kwargs["target_excess_mask"],
            )
        else:
            loss_excess = torch.tensor(0.0)

        # Combine losses
        loss = loss_mix + loss_excess

        # Apply transform
        y_linear = self.model.transform.forward(y_linear)
        y_excess = self.model.transform.std * y_excess
        y = y_linear + y_excess

        return y, loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self.forward_loss(**batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
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
        preds, loss = self.forward_loss(**batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
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
        preds, loss = self.forward_loss(**batch)
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
        learnable_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
                continue
            learnable_params.append(param)
        # if not self.freeze_encoder:
        #     learnable_params = chain(learnable_params, self.encoder.parameters())

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer


if __name__ == "__main__":
    import logging
    from datetime import timedelta

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from electrolyte_fm.models.excess_physics_model import ExcessPhysicsConfig
    from electrolyte_fm.utils.lr_schedule import RelativeCosineWarmup

    from ..data_modules.mixture_dataset import ComponentDataModule

    logging.basicConfig(level=logging.INFO)

    val_loss_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={val/loss:.3f}",
        monitor="val/loss",
        save_top_k=2,
        verbose=True,
        save_last="link",
        enable_version_counter=True,
        auto_insert_metric_name=False,
        save_weights_only=True,
    )
    val_loss_ckpt.CHECKPOINT_NAME_LAST = "best"
    step_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}",
        monitor="step",
        verbose=True,
        mode="max",
        save_top_k=2,
        save_last=True,
        train_time_interval=timedelta(minutes=15),
        auto_insert_metric_name=False,
    )
    step_ckpt.CHECKPOINT_NAME_LAST = "last"

    config = {
        "data": {
            "path": "excess_dataset",
            "batch_size": 16,
        },
        "trainer": {
            "lr": 1e-3,
            "num_training_steps": 10_000,
        },
    }
    model = ExcessPhysicsModel.from_pretrained_encoder(
        name_or_path="models/mist-ti624ev1",
        temperature_dependence=[-1, 1, 2],
        num_control=2,
        interactions="square-difference",
    )
    config["model"] = model.config.to_dict()

    config["data"]["target_columns"] = model.config.target_columns
    dm = ComponentDataModule(**config["data"])

    lit = ExcessPhysicsLightningModel(
        model=model,
        optimizer=lambda params: torch.optim.AdamW(params, lr=config["trainer"]["lr"]),
        lr_schedule=lambda optimizer: RelativeCosineWarmup(
            optimizer,
            num_training_steps=int(config["trainer"]["num_training_steps"]),
            num_warmup_steps="beta2",
        ),
    )

    trainer = Trainer(
        precision=32,
        callbacks=[val_loss_ckpt, step_ckpt, LearningRateMonitor(), ModelCheckpoint()],
        logger=WandbLogger(project="excess_physics"),
        enable_progress_bar=False,
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(lit, datamodule=dm)
