# Configure TLS before model/tokenizer clients are imported by the CLI.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC

from electrolyte_fm.models.model_utils import create_position_ids
from electrolyte_fm.utils.cli import MistLightningCLI

logger = logging.getLogger(__name__)


def per_layer_probe(
    hidden_size: int, features: int, n_layers: int, location: str = "output"
) -> dict[str, nn.Module]:
    probes = {}
    template: str = "*encoder.layer.{layer}.{location}"
    for layer in range(n_layers):
        hook_name = template.format(layer=layer, location=location)
        probes[hook_name] = nn.Linear(hidden_size, features)
    return probes


def probe_everything(
    hidden_size: int,
    features: int,
    n_layers: int,
    intermediate_size: int | None = None,
):
    probes = {}
    intermediate_size = intermediate_size or hidden_size
    for location, size in [
        ("output", hidden_size),
        ("intermediate", intermediate_size),
        ("attention", hidden_size),
        ("output.dense", hidden_size),
    ]:
        probes.update(per_layer_probe(size, features, n_layers, location))
    return probes


ProbeConfigCallable = Callable[..., dict[str, nn.Module]]


class LightningProbe(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        probes: ProbeConfigCallable | None = None,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()

        self.model = model.requires_grad_(False)
        if probes is None:
            config = self.model.config
            probes = probe_everything(
                config.hidden_size,
                5,
                config.num_hidden_layers,
                config.intermediate_size,
            )
        elif callable(probes):
            probes = probes()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        # The CLI records constructor descriptions; never pickle the frozen
        # encoder or another copy of the probes into every checkpoint.
        self.save_hyperparameters(ignore=["model", "probes"])
        self.hookpoints = self._identify_hookpoints(probes.keys())
        assert len(self.hookpoints) > 0

        # Setup probes
        self._probes = nn.ModuleList(probes.values())
        self._probe_points = list(probes.keys())
        self._hooks_installed = {}
        self._activations = {}

        # Add metrics
        self.val_metrics = nn.ModuleList(
            AUROC(task="binary", thresholds=100) for probe in self._probe_points
        )
        print(self.val_metrics)

        # Don't error due to missing model weights
        self.strict_loading = False

    def state_dict(self):
        # Don't save the model, it is not being trained
        return {k: v for k, v in super().state_dict().items() if "model" not in k}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        # Remove hooks from model (not picklable))
        for hook in state_dict.pop("_hooks_installed", {}).values():
            hook.remove()
        state_dict.pop("_hooks_installed", None)
        state_dict["_prob_points"] = self._probe_points
        state_dict["hookpoints"] = self.hookpoints
        checkpoint["state_dict"] = state_dict

        # Don't save activations
        state_dict.pop("_activations", None)
        checkpoint["probe_metrics"] = {
            key: float(value)
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("val/")
        }

    def named_probes(self):
        yield from zip(self._probe_points, self._probes)

    def _identify_hookpoints(self, probe_locs: Iterable[str]):
        hooks: set[str] = set()
        for name, _ in self.model.named_modules():
            for loc in probe_locs:
                if fnmatchcase(name, loc):
                    hooks.add(name)

        hooks = list(hooks)
        hooks.sort()
        return hooks

    def _install_hooks(self):
        for name in self.hookpoints:
            if name not in self._hooks_installed:
                hook = self._create_act_hook(name, self._activations)
                self._hooks_installed[name] = self.model.get_submodule(
                    name
                ).register_forward_hook(hook)
                logger.debug("Installed hook for %s", name)

        return self._activations

    @staticmethod
    def _create_act_hook(name: str, results: dict):
        def hook(module: nn.Module, input, output: torch.Tensor):
            if isinstance(output, tuple):
                output = output[0]
            assert isinstance(output, torch.Tensor)
            results[name] = output[:, 0, :].detach()

        return hook

    def forward(self, *args, **kwargs):
        self.model.eval()
        activations = self._install_hooks()
        with torch.no_grad():
            self._encode(*args, **kwargs)
        out = defaultdict(dict)
        for name, probe in self.named_probes():
            for hook in self.hookpoints:
                if fnmatchcase(hook, name):
                    act = activations[hook]
                    out[name][hook] = probe(act)

        return out

    def _encode(self, input_ids, **kwargs):
        config = getattr(self.model, "config", None)
        if "position_ids" not in kwargs and hasattr(config, "position_padding_idx"):
            kwargs["position_ids"] = create_position_ids(
                input_ids, config.pad_token_id, config.position_padding_idx
            )
        return self.model(input_ids, **kwargs)

    def forward_fit(self, batch: dict):
        self.model.eval()
        activations = self._install_hooks()
        target = batch.pop("target")
        self._encode(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = []
        out = {}
        for name, probe in self.named_probes():
            probe_loss = []
            for hook in self.hookpoints:
                if fnmatchcase(hook, name):
                    act = activations[hook].detach()
                    y = probe(act)
                    probe_loss.append(
                        F.binary_cross_entropy_with_logits(y, target.to(dtype=y.dtype))
                    )

            out[f"{name}-probe-loss"] = sum(probe_loss)
            loss.append(sum(probe_loss))

        out["loss"] = sum(loss) / len(loss)
        return out

    def training_step(self, batch):
        out = self.forward_fit(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in out.items()}, on_step=False, on_epoch=True
        )
        return out["loss"]

    def validation_step(self, batch):
        target = batch.pop("target")
        out = self.forward(batch["input_ids"], attention_mask=batch["attention_mask"])
        metrics = {}
        loss = []
        for probe, probe_metrics in zip(self._probe_points, self.val_metrics):
            probe_pred = []
            for hook in self.hookpoints:
                if fnmatchcase(hook, probe):
                    probe_pred.append(out[probe][hook])

            probe_pred = torch.stack(probe_pred)
            probe_pred = probe_pred.view(-1, probe_pred.shape[-1])
            probe_loss = F.binary_cross_entropy_with_logits(
                probe_pred,
                target.to(dtype=probe_pred.dtype),
            )
            probe_metrics.update(probe_pred, target)
            metrics[f"val/{probe}-loss"] = probe_loss
            metrics[f"val/{probe}-auroc"] = probe_metrics
            loss.append(probe_loss)

        metrics["val/loss"] = sum(loss) / len(loss)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return metrics["val/loss"]

    def configure_optimizers(self):
        optimizer = self.optimizer(self._probes.parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer


def encoder_from_finetuned(
    name_or_path: str, tokenizer: str | None = None
) -> nn.Module:
    """Reuse the production loader for released and Lightning task checkpoints."""
    from .prod_finetune import load_finetuned

    return load_finetuned(name_or_path, tokenizer=tokenizer).encoder


def cli_main(args=None):
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    cli = MistLightningCLI(
        LightningProbe,
        args=args,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "logger": lazy_instance(
                WandbLogger, project="linear-probes", save_code=True
            ),
            "max_epochs": 1000,
            "precision": (
                "bf16-mixed"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "32-true"
            ),
        },
        parser_kwargs={"parser_mode": "jsonnet"},
        run=False,
    )
    trainer: pl.Trainer = cli.trainer
    model: LightningProbe = cli.model
    # Retain the loader description for provenance, never the encoder object.
    model.hparams["model"] = cli.config_dump["model"]["model"]
    # Replace Lightning's implicit aggregate checkpoint with one per probe.
    trainer.callbacks = [
        c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)
    ]
    ckpts = []
    for probe, _ in model.named_probes():
        probe_name = probe.replace(".", "-").replace("*", "star")
        monitor = f"val/{probe}-loss"
        auroc = f"val/{probe}-auroc"
        ckpts.append(
            ModelCheckpoint(
                dirpath=Path(trainer.default_root_dir) / "checkpoints",
                monitor=monitor,
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
                auto_insert_metric_name=False,
                filename=probe_name
                + "--epoch-{epoch}--loss-{"
                + monitor
                + ":.3f}--auroc-{"
                + auroc
                + ":.3f}",
            )
        )
    trainer.callbacks.extend(ckpts)

    trainer.fit(model, cli.datamodule)
    if not all(callback.best_model_path for callback in ckpts):
        raise RuntimeError(
            "Training finished without a validated checkpoint for every probe"
        )
    # Small run metadata only; probe tensors stay in Lightning checkpoints and
    # the existing Julia loader writes their analysis to JLD2.
    torch.save(
        {
            "training_steps": trainer.global_step,
            "requested_steps": trainer.max_steps,
            "step_budget_reached": trainer.max_steps > 0
            and trainer.global_step >= trainer.max_steps,
        },
        Path(trainer.default_root_dir) / "fit.pt",
    )
    return cli


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli_main()
