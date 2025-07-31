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

from torchmetrics import MetricCollection
from ..utils.metrics import (
    get_metric,
    masked_loss,
)
from .model_utils import masked_mean_pool, sparsity_weights
from .normalize import Standardize
from .polynomials import LagrangePolynomial
from .physics_task_heads import ArrtheniusActivation, LinearExogenousEffect
from ..utils.progressive_thawing import ProgressiveThawing


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
    temperature_dependence: str | None = None
    relative_excess: bool = False
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
    elif name == "concat":
        return ConcatFusion(*args, **kwargs)
    else:
        raise ValueError(f"Unknown fusion: {name}")


class PairwiseInteraction(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_targets: int = 1,
        dropout: float = 0.1,
        n_env: int = 0,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_targets = n_targets
        self.n_env = n_env
        self.mlp_emb = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_in + n_env, n_in),
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

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, e: torch.Tensor | None = None
    ) -> torch.Tensor:
        a = self.mlp_emb(a)
        b = self.mlp_emb(b)
        d = self.distance(a, b)
        if self.n_env > 0:
            d = torch.cat([d, e], dim=-1)
        print("Distance", d.shape)
        y = self.mlp(d)
        y = y.reshape(*y.shape[:-1], self.n_targets, self.n_out)
        return y + y.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return (a - b).pow(2)


class GaussianFusion(PairwiseInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = -1 * (a - b).pow(2)
        return dist.exp()


class EquivariantInteraction(PairwiseInteraction):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, e: torch.Tensor | None = None
    ) -> torch.Tensor:
        a = self.mlp_emb(a)
        b = self.mlp_emb(b)
        emb_a, emb_b = self.distance(a, b)
        if self.n_env > 0:
            emb_a = torch.cat([emb_a, e], dim=-1)
            emb_b = torch.cat([emb_b, e], dim=-1)
        y_a = self.mlp(emb_a).reshape(*emb_a.shape[:-1], self.n_targets, self.n_out)
        y_b = self.mlp(emb_b).reshape(*emb_b.shape[:-1], self.n_targets, self.n_out)
        return y_a + y_b.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return a - b, b - a


class SoftmaxFusion(EquivariantInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor):
        y = torch.stack([a, b]).log_softmax(dim=0)
        return y[0], y[1]


class ConcatFusion(EquivariantInteraction):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        super().__init__(n_in, n_out, **kwargs)
        self.mlp[0] = nn.Linear(2 * self.n_in + self.n_env, self.n_in)
        self.reset_parameters()

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return torch.cat([a, b], dim=-1), torch.cat([b, a], dim=-1)


class ExcessPhysicsModel(nn.Module):
    def __init__(self, config: ExcessPhysicsConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_config(config.encoder)

        # Configure Pairwise interaction model
        n_env = 0
        n_temperature_targets = 1
        if config.temperature_dependence == "arrhenius":
            n_temperature_targets = 2
            self.temperature_dependence = ArrtheniusActivation()
        elif config.temperature_dependence == "concat":
            n_env = 1
            self.temperature_dependence = lambda x, t: x
        elif config.temperature_dependence == "locally-linear":
            n_temperature_targets = 2
            n_env = 1
            self.temperature_dependence = LinearExogenousEffect()

        else:
            self.temperature_dependence = lambda x, t: x

        self.pairwise_interaction = pairwise_fusion(
            config.interactions,
            config.encoder.hidden_size,
            config.num_control * n_temperature_targets,
            n_targets=config.num_targets,
            n_env=n_env,
        )
        self.component_properties = nn.Sequential(
            nn.Linear(config.encoder.hidden_size + n_env, config.encoder.hidden_size),
            nn.Dropout(config.dropout),
            nn.SiLU(),
            nn.Linear(
                config.encoder.hidden_size,
                config.num_targets * n_temperature_targets,
            ),
        )
        self.excess_polynomial = LagrangePolynomial(
            polynomial_order=self.config.num_control + 2,
            zero_endpoints=True,
        )
        self.transform = Standardize(num_outputs=config.num_targets)
        self.excess_transform = Standardize(num_outputs=config.num_targets)

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
        t_ij = temperature.view(B, 1, 1).expand(-1, I, -1).reshape(B * I, 1)
        pw_coeffs = self.pairwise_interaction(e_i, e_j, t_ij)  # (B*I, E) -> (B*I, T, P)
        T = self.pairwise_interaction.n_targets
        P = self.pairwise_interaction.n_out
        assert pw_coeffs.shape == (B * I, T, P)

        # Compute partial pairwise concentrations
        # I.e. Partial concentration of i assuming an i+j mixture
        x_t = composition[:, indices[0]] + composition[:, indices[1]]
        x_t = x_t.clamp(min=0, max=1)
        x_i = composition[:, indices[0]] / x_t
        x_i = torch.where(x_i.abs() > 1e-8, x_i, torch.tensor(0.0).to(x_i))
        x_i = x_i.clamp(min=0, max=1)
        x_i = x_i.view(B * I, 1)

        # Apply temperature dependence to coefficients
        pw_coeffs = self.temperature_dependence(pw_coeffs, t_ij.view(B * I, 1, 1))

        # Evaluate interaction polynomials at compositions
        pw = x_t.view(B * I, 1) * self.excess_polynomial(pw_coeffs, x_i)
        assert pw.shape == (B * I, T)

        # Sum over interactions to get excess properties
        y_excess = pw.reshape(B, I, -1).sum(dim=1)
        assert y_excess.shape == (B, T)

        # Predict Pure Property (B, C, E) -> (B, C, T)
        if self.config.temperature_dependence in ["concat", "locally-linear"]:
            t_e = temperature.view(B, 1, 1).expand(-1, C, -1)
            y_target = self.component_properties(torch.cat([embs, t_e], dim=-1))
        else:
            y_target = self.component_properties(embs)

        y_target = self.temperature_dependence(y_target, temperature.view(B, 1, 1))
        assert y_target.shape == (B, C, T), f"Got: {y_target.shape}, vs {(B, C, T)}"

        # Linear Mixing (B, C, T) -> (B, T)
        assert composition.shape == (B, C)
        y_linear = (y_target * composition.view(B, C, 1)).sum(dim=1)
        assert y_linear.shape == (B, T)

        if self.config.relative_excess:
            y_excess *= y_linear

        # Transform to real-units
        y_linear = self.transform.forward(y_linear)
        y_excess = self.excess_transform.forward(y_excess)
        y = y_linear + y_excess

        return y, y_linear, y_excess


def clean_target_name(target: str) -> str:
    return target.split("[")[0].strip().replace(" ", "_")


class ExcessPhysicsLightningModel(LightningModule):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        model: ExcessPhysicsModel | ExcessPhysicsConfig,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        sparsity_weighted_loss: bool = False,
        metrics: list[str] = ["rmse", "mae", "r2"],
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

        self.sparsity_weighted_loss = sparsity_weighted_loss
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lossfn = nn.MSELoss(reduction="none")
        self.target_metrics = metrics

        mc = {}
        for target in self.model.config.target_columns:
            target = clean_target_name(target)
            for metric in metrics:
                mc[f"{target}/{metric}"] = get_metric(metric, "regression")
                mc[f"excess_{target}/{metric}"] = get_metric(metric, "regression")

        metrics = MetricCollection(mc)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Sparsity Weights
        self.target_weight: torch.Tensor | None = None
        self.excess_weight: torch.Tensor | None = None

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["", "_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min,last")

    def on_fit_start(self):
        """Standardized training data"""
        state = None
        sparsity = None
        if self.global_rank == 0:
            assert self.trainer.datamodule.target_dataset is not None
            ds = self.trainer.datamodule.target_dataset
            state = self.model.transform.fit(ds)
            state_excess = self.model.excess_transform.fit(ds, name="target_excess")
            state_excess["mean"].zero_()

            # Compute target weights
            sparsity = sparsity_weights(ds, ["target_mask", "target_excess_mask"])

        # Broadcast normalization state
        state = self.trainer.strategy.broadcast(state)
        state_excess = self.trainer.strategy.broadcast(state_excess)
        self.model.transform.load_state_dict(state)
        self.model.excess_transform.load_state_dict(state_excess)

        # Broadcast sparsity weights
        sparsity = self.trainer.strategy.broadcast(sparsity)
        assert sparsity is not None and isinstance(sparsity, dict)
        self.target_weight = sparsity["target_mask"]
        self.excess_weight = sparsity["target_excess_mask"]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_loss(self, **kwargs):
        y, y_linear, y_excess = self.forward(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            temperature=kwargs["temperature"],
            composition=kwargs["composition"],
        )

        # Compute standardized losses
        loss_mix = masked_loss(
            self.lossfn,
            self.model.transform.inverse(y),
            self.model.transform.inverse(kwargs["target"]),
            kwargs["target_mask"],
            weight=self.target_sparsity.view(1, -1)
            if self.sparsity_weighted_loss
            else None,
        )
        if "target_excess" in kwargs:
            loss_excess = masked_loss(
                self.lossfn,
                self.model.excess_transform.inverse(y_excess),
                self.model.excess_transform.inverse(kwargs["target_excess"]),
                kwargs["target_excess_mask"],
                weight=self.excess_sparsity.view(1, -1)
                if self.sparsity_weighted_loss
                else None,
            )
        else:
            loss_excess = torch.tensor(0.0)

        loss = loss_mix + loss_excess

        return (y, y_linear, y_excess), loss

    def update_metrics(self, preds, batch, metrics):
        y, _, y_excess = preds  # Only y is used
        targets = self.model.config.target_columns
        self._masked_metric(
            metrics,
            y,
            batch["target"],
            batch["target_mask"],
            (clean_target_name(t) for t in targets),
        )
        self._masked_metric(
            metrics,
            y_excess,
            batch["target_excess"],
            batch["target_excess_mask"],
            ("excess_" + clean_target_name(t) for t in targets),
        )

    def _masked_metric(self, metrics, y, y_ref, mask, targets):
        for idx, target in enumerate(targets):
            y_pred = y[:, idx]
            y_true = y_ref[:, idx]
            target_mask = mask[:, idx]

            if not target_mask.any():
                continue

            y_pred = y_pred[target_mask]
            y_true = y_true[target_mask]
            for metric in self.target_metrics:
                metrics[f"{target}/{metric}"].update(y_pred, y_true)

    def stage_step(self, stage: str, batch):
        preds, loss = self.forward_loss(**batch)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        metrics = getattr(self, f"{stage}_metrics")
        self.update_metrics(preds, batch, metrics)
        self.log_dict(metrics.compute(), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("train", batch)

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("val", batch)

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("test", batch)

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
    from argparse import ArgumentParser

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from electrolyte_fm.models.excess_physics_model import ExcessPhysicsConfig
    from electrolyte_fm.utils.lr_schedule import RelativeCosineWarmup

    from ..data_modules.mixture_dataset import ComponentDataModule

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, required=False)
    parser.add_argument("--config-patch", type=str, default=None, required=False)
    args = parser.parse_args()

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
        "model": {
            "name_or_path": "models/mist-ti624ev1",
            "temperature_dependence": "concat",
            "num_control": 2,
            "interactions": "difference",
        },
        "data": {
            "path": "excess_dataset_v5/random",
            "batch_size": 16,
            "randomize": True,
        },
        "trainer": {
            "lr": 1e-3,
            "num_training_steps": 10_000,
            "freeze": {
                "initial": ["model.encoder"],
                "thaw_embeddings": True,
                "stage_duration": 1,
                "thaw_depth": 2,
            },
        },
    }
    if args.config:
        config = json.loads(Path(args.config).read_text())
        if "fit" in config:
            config = config["fit"]

    if args.config_patch:
        from electrolyte_fm.utils.cli import recursive_update

        patch = json.loads(Path(args.config_patch).read_text())
        config = recursive_update(config, patch)

    model = ExcessPhysicsModel.from_pretrained_encoder(**config["model"])
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
    lit.compile()

    # Construct Thawing Schedule
    if config["trainer"]["freeze"] == "encoder":
        freeze = ProgressiveThawing(initial=["model.encoder"], stages=[])
    elif config["trainer"].get("freeze", None) is None:
        freeze = ProgressiveThawing(initial=[], stages=[])
    else:
        freeze_config = config["trainer"]["freeze"]
        last_layer = model.config.encoder.num_hidden_layers - 1
        thaw_depth = freeze_config["thaw_depth"]
        max_thaw = last_layer - thaw_depth if thaw_depth > 0 else 0
        stages = [
            [f"model.encoder.encoder.layer.{i}"]
            for i in range(last_layer, max_thaw, -1)
        ]
        if freeze_config["thaw_embeddings"]:
            stages = [["model.encoder.embeddings"], *stages]

        freeze = ProgressiveThawing(initial=freeze_config["initial"], stages=stages)

    trainer = Trainer(
        precision=32,
        callbacks=[
            val_loss_ckpt,
            step_ckpt,
            LearningRateMonitor(),
            ModelCheckpoint(),
            freeze,
        ],
        logger=WandbLogger(project="excess_physics"),
        enable_progress_bar=False,
        max_steps=int(config["trainer"]["num_training_steps"]),
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(lit, datamodule=dm)
