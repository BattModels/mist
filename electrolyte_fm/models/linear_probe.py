import logging
from typing import Iterable, Dict, Any, Callable, Optional
from collections import defaultdict
from jsonargparse import lazy_instance
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from fnmatch import fnmatchcase
from torchmetrics import AUROC


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
    intermediate_size: Optional[int] = None,
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


ProbeConfigCallable = Callable[Any, dict[str, nn.Module]]


class LightningProbe(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        probes: ProbeConfigCallable = per_layer_probe,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()

        self.model = model.requires_grad_(False)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters()
        self.hookpoints = self._identify_hookpoints(probes.keys())
        assert len(self.hookpoints) > 0

        # Setup probes
        self._probes = nn.ModuleList(probes.values())
        self._probe_points = list(probes.keys())
        self._hooks_installed = dict()
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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
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
                logging.debug(f"Installed hook for %s", name)

        return self._activations

    @staticmethod
    def _create_act_hook(name: str, results: dict):
        def hook(module: nn.Module, input, output: torch.Tensor):
            if isinstance(output, tuple):
                output = output[0]
            assert isinstance(output, torch.Tensor)
            results[name] = output[:, 0, :].detach()
            return None

        return hook

    def forward(self, *args, **kwargs):
        self.model.eval()
        activations = self._install_hooks()
        with torch.no_grad():
            self.model(*args, **kwargs)
        out = defaultdict(dict)
        for name, probe in self.named_probes():
            for hook in self.hookpoints:
                if fnmatchcase(hook, name):
                    act = activations[hook]
                    out[name][hook] = probe(act)

        return out

    def forward_fit(self, batch: dict):
        self.model.eval()
        activations = self._install_hooks()
        target = batch.pop("target")
        self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
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


if __name__ == "__main__":
    import smirk
    import json
    from jsonargparse import lazy_instance
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from electrolyte_fm.utils.cli import MistLightningCLI

    logging.basicConfig(level=logging.INFO)

    def encoder_from_finetuned(name_or_path: str) -> nn.Module:
        from .prod_finetune import MISTFinetuned

        model = MISTFinetuned.from_pretrained(name_or_path)
        print(model.encoder)
        return model.encoder

    cli = MistLightningCLI(
        LightningProbe,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "logger": lazy_instance(
                WandbLogger, project="linear-probes", save_code=True
            ),
            "max_epochs": 1000,
        },
        parser_kwargs={"parser_mode": "jsonnet"},
        run=False,
    )
    trainer: pl.Trainer = cli.trainer
    model: LightningProbe = cli.model
    ckpts = []
    for probe, _ in model.named_probes():
        probe_name = probe.replace(".", "-").replace("*", "star")
        monitor = f"val/{probe}-loss"
        auroc = f"val/{probe}-auroc"
        ckpts.append(
            ModelCheckpoint(
                monitor=monitor,
                save_top_k=1,
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
