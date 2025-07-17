import torch
from accelerate import Accelerator
from pathlib import Path

from smirk import SmirkTokenizerFast
from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsLightningModel,
    ExcessPhysicsModel,
)
from electrolyte_fm.data_modules.mixture_dataset import ComponentDataModule
from transformers import DataCollatorWithPadding
from torch.utils.data import IterableDataset, default_collate


class MixtureDataset(IterableDataset):
    def __init__(self, mixtures: list[dict], tokenizer=None, n: int = 50):
        self.tokenizer = tokenizer or SmirkTokenizerFast()
        self.token_collate = DataCollatorWithPadding(self.tokenizer)
        self.mixtures = mixtures
        self.n = n

    def __iter__(self):
        for mixture in self.mixtures:
            compounds = mixture["compounds"]
            temperature = mixture.get("temperature", 293.15)
            assert len(compounds) == 2
            o = self.token_collate(self.tokenizer(compounds))

            # Sample over compositions
            for x1 in torch.linspace(0, 1, steps=self.n):
                yield {
                    "compounds": compounds,
                    "composition": torch.tensor([x1, 1 - x1]),
                    "input_ids": o["input_ids"],
                    "attention_mask": o["attention_mask"],
                    "composition": torch.tensor([x1, 1 - x1]),
                    "temperature": torch.tensor(temperature),
                }

    def get_dataloader(self, batch_size=16):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def collate_fn(self, batch):
        out = default_collate(
            [{k: obs[k] for k in batch[0].keys() if k != "compounds"} for obs in batch]
        )
        out["compounds"] = [obs["compounds"] for obs in batch]
        return out


def load_excess_model(ckpt):
    return ExcessPhysicsLightningModel.load_from_checkpoint(ckpt).model


def evaluate_mixtures(model: ExcessPhysicsModel, mixtures: list[dict]):
    ds = MixtureDataset(mixtures)
    dl = ds.get_dataloader()
    return evaluate(model, dl)


def evaluate_dataset(model: ExcessPhysicsModel, path):
    assert Path(path).exists()
    dm = ComponentDataModule(
        path,
        batch_size=16,
        target_columns=model.config.target_columns,
    )
    dm.prepare_data()
    dm.setup("fit")

    return evaluate(model, dm.train_dataloader())


def evaluate(model: ExcessPhysicsModel, dataloader):
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare_model(model.eval(), device_placement=True)

    for batch in dataloader:
        smi_columns = [
            k for k in batch.keys() if not isinstance(batch[k][0], torch.Tensor)
        ]
        with torch.no_grad():
            y, y_linear, y_excess = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                temperature=batch["temperature"].to(device),
                composition=batch["composition"].to(device),
            )

        for bdx in range(batch["input_ids"].shape[0]):
            out = {
                **{k: batch[k][bdx] for k in smi_columns},
                "temperature": batch["temperature"][bdx].item(),
                "composition": batch["composition"][bdx].tolist(),
                "y": y[bdx].tolist(),
                "y_excess": y_excess[bdx].tolist(),
                "y_linear": y_linear[bdx].tolist(),
            }
            if "target" in batch:
                out["target"] = batch["target"][bdx].tolist()
                out["target_mask"] = batch["target_mask"][bdx].tolist()
            if "target_excess" in batch:
                out["target_excess"] = batch["target_excess"][bdx].tolist()
                out["target_excess_mask"] = batch["target_excess_mask"][bdx].tolist()

            yield out
