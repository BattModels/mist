from typing import Iterable
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from fnmatch import fnmatchcase


class LightningProbe(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        probes: dict[str, nn.Module],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.hookpoints = self._identify_hookpoints(probes.keys())
        self._probes = nn.ModuleList(probes.values())
        self._probe_points = list(probes.keys())
        self.activations = dict()

    @property
    def probes(self):
        yield from zip(self._probe_points, self._probes)

    def _identify_hookpoints(self, hooks: Iterable[str]):
        hooks: set[str] = set()
        for name, _ in self.model.named_modules():
            if any((fnmatchcase(name, hook) for hook in hooks)):
                hooks.add(name)

        hooks = list(hooks)
        hooks.sort()
        return hooks

    def _install_hooks(self, results: dict[str, torch.Tensor]):
        for name in self.hookpoints:

            def hook(module: nn.Module, input, output: torch.Tensor):
                assert isinstance(output, torch.Tensor)
                results[name] = output
                return None

            self.model.get_submodule(name).register_forward_hook(hook)

    def on_train_start(self):
        self.model.to(self.device)
        self._install_hooks(self.activations)

    def forward(self, batch: dict):
        self.model.eval()
        with torch.no_grad():
            self.model(batch)
        out = defaultdict(dict)
        for name, probe in self.probs.items():
            for hook in self.hookpoints:
                if fnmatchcase(name, hook):
                    act = self.activations[hook]
                    out[name][hook] = probe(act)

        return out

    def forward_fit(self, batch: dict):
        self.model.eval()
        target = batch.pop("probe_target")
        self.model(**batch)
        loss = torch.tensor(0.0)
        out = {}
        for name, probe in self.probs.items():
            probe_loss = torch.tensor(0.0)
            for hook in self.hookpoints:
                if fnmatchcase(name, hook):
                    act = self.activations[hook]
                    y = probe(act)
                    probe_loss += F.binary_cross_entropy_with_logits(y, target)

            out[f"{name}-probe-loss"] = probe_loss
            loss += probe_loss

        out["loss"] = loss
        return out

    def training_step(self, batch):
        out = self.forward_fit(batch)
        self.log_dict({f"train/{k}": v for k, v in out.items()})
        return out["loss"]

    def validation_step(self, batch):
        out = self.forward_fit(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in out.items()}, on_step=False, on_epoch=True
        )
        return out["loss"]

    def configure_optimizers(self):
        optimizer = self.optimizer(self._probes.parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM
    from ..data_modules.lipinski_dataset import LipinskiDataModule
    from lightning.pytorch import Trainer

    name_or_path = "ibm/MoLFormer-XL-both-10pct"
    model = AutoModelForMaskedLM.from_pretrained(name_or_path, trust_remote_code=True)
    dm = LipinskiDataModule(
        name_or_path="hiv",
        tokenizer=name_or_path,
        encoding="smiles-canonical",
        num_workers=4,
    )
    hidden_size = model.config.hidden_size
    probes = {
        f"*.encoder.layer.{layer}.output": nn.Linear(hidden_size, 5)
        for layer in range(hidden_size)
    }
    lm = LightningProbe(model, probes)

    trainer = Trainer()
    trainer.fit(lm, datamodule=dm)
