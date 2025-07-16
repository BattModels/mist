from pathlib import Path

import torch
import typer
from datasets import Dataset, IterableDatasetDict, load_dataset
from torch.utils.data import default_collate

from .utils import MolEncoding, stack_columns
from .property_prediction_dataset import PropertyPredictionDataModule, collate_target

cli = typer.Typer()


class ComponentDataModule(PropertyPredictionDataModule):
    def __init__(
        self,
        path: str | Path,
        n_components: int = 2,
        include_temperature: bool | str = True,
        smi_column: str = "smi{:d}",
        x_column: str = "x{:d}",
        excess_columns: str | list[str] | None = "excess {:s}",
        **kwargs,
    ):
        self.path = Path(path)
        self.include_temperature = bool(include_temperature)
        self.n_components = int(n_components)
        self.smi_columns = [smi_column.format(n + 1) for n in range(self.n_components)]
        self.x_columns = [x_column.format(n + 1) for n in range(self.n_components)]
        assert self.path.exists()
        super().__init__(smi_column=smi_column, **kwargs)
        assert len(self.target_columns) >= 1
        assert len(self.smi_columns) == len(self.x_columns) == self.n_components

        if isinstance(excess_columns, str):
            self.excess_columns = [
                excess_columns.format(col) for col in self.target_columns
            ]
        elif isinstance(excess_columns, list):
            self.excess_columns = excess_columns
            assert len(self.excess_columns) == len(self.target_columns)
        else:
            self.excess_columns = None

    def _get_dataset(self):
        ds = load_dataset(
            "arrow",
            name=self.path.name,
            data_files={
                "train": str(self.path.joinpath("train/*.arrow")),
                "validation": str(self.path.joinpath("validation/*.arrow")),
                "test": str(self.path.joinpath("test/*.arrow")),
            },
            keep_in_memory=False,
            streaming=True,
            save_infos=False,
        )
        assert isinstance(ds, IterableDatasetDict)
        return ds

    def setup(self, stage: str) -> None:
        # Prepare target columns
        ds = self.dataset
        ds = ds.map(
            collate_target,
            batched=False,
            fn_kwargs={"target_columns": self.target_columns},
            remove_columns=self.target_columns,
        )

        if self.excess_columns:
            ds = ds.map(
                collate_target,
                batched=False,
                fn_kwargs={
                    "target_columns": self.excess_columns,
                    "name": "target_excess",
                },
                remove_columns=self.excess_columns,
            )

        ds = ds.map(
            stack_columns,
            batched=True,
            fn_kwargs={"columns": self.x_columns, "output": "composition"},
            remove_columns=self.x_columns,
        )

        # Maybe encode and tokenize
        if not self.randomize:
            ds = ds.map(
                encode_and_tokenize_mixture,
                batched=True,
                fn_kwargs={
                    "smi_columns": self.smi_columns,
                    "randomize": self.randomize,
                    "encoding": self.encoding,
                    "tokenizer": self.tokenize,
                    "token_collator": self.token_collator,
                },
            )

        # Filter to input columns
        columns = ["target", "target_mask", "composition"]
        if self.include_encoding or self.randomize:
            columns.extend(self.smi_columns)
        if self.excess_columns:
            columns.extend(["target_excess", "target_excess_mask"])

        # Filter to complete entries
        def has_training_data(x):
            n_targets = x["target_mask"].sum() > 0
            n_excess = x["target_excess_mask"].sum() > 0
            return n_targets or n_excess

        ds = ds.filter(has_training_data, batched=False)

        if not self.randomize:
            columns.extend(["input_ids", "attention_mask"])
        if self.include_temperature:
            ds = ds.rename_column("temperature [kelvin]", "temperature")
            columns.append("temperature")
        ds = ds.select_columns(columns)

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]

        # Dataset for normalization
        norm_columns = ["target", "target_mask", "temperature"]
        if self.excess_columns:
            norm_columns.extend(["target_excess", "target_excess_mask"])
        self.target_dataset = ds["train"].select_columns(norm_columns)

    def collate_fn(self, batch):
        if self.include_encoding:
            compounds = [
                [batch[bdx][k] for k in self.smi_columns] for bdx in range(len(batch))
            ]
            batch = {
                k: default_collate([obs[k] for obs in batch])
                for k in batch[0].keys()
                if k not in self.smi_columns
            }
            batch["compounds"] = compounds
        else:
            batch = default_collate(batch)

        batch["composition"] = default_collate(batch["composition"]).T
        if self.randomize:
            batch = encode_and_tokenize_mixture(
                batch,
                tokenizer=self.tokenize,
                encoding=self.encoding,
                smi_columns=self.smi_columns,
                randomize=self.randomize,
                token_collator=self.token_collator,
            )
            if not self.include_encoding:
                for smi in self.smi_columns:
                    batch.pop(smi)

        for k in ["target", "target_excess", "temperature", "composition"]:
            if not isinstance(batch[k], torch.Tensor):
                continue

            if batch[k].dtype == torch.float64:
                batch[k] = batch[k].to(torch.float32)

        return batch


def encode_and_tokenize_mixture(
    batch: dict[str, list | torch.Tensor],
    tokenizer=None,
    encoding: MolEncoding = MolEncoding.SMILES,
    smi_columns: list[str] = [],
    randomize: bool = True,
    token_collator=None,
):
    encode = encoding.random if randomize else encoding
    B = len(batch[smi_columns[0]])
    smi = [encode(batch[col][bdx]) for bdx in range(B) for col in smi_columns]  # (B*N)

    toks = token_collator(tokenizer(smi))
    for k in ["input_ids", "attention_mask"]:
        batch[k] = torch.reshape(toks[k], (B, len(smi_columns), -1))

    return batch


@cli.command()
def split_dataset(data: Path, output: Path, split: str = "random", split_idx: int = 0):
    from .molnet_dataset import train_val_test_split
    from .splits import (
        StrictEntityHoldoutSplitter,
        EntityHoldoutSplitter,
        apply_splitter,
    )
    from sklearn.model_selection import GroupShuffleSplit

    ds: Dataset = load_dataset("csv", data_files=[str(data)], split="train")
    ds = ds.map(lambda x: {"x2": 1 - x["x1"]}, batched=False)
    if split == "random":
        ds = train_val_test_split(ds)

    elif split == "k-compound":
        ds = apply_splitter(
            ds,
            EntityHoldoutSplitter(entity_cols=["smi1", "smi2"]),
            split=split_idx,
        )
    elif split == "k-compound-strict":
        ds = apply_splitter(
            ds,
            StrictEntityHoldoutSplitter(entity_cols=["smi1", "smi2"]),
            split=split_idx,
        )
    elif split == "k-mixture":

        def label_mixture(x):
            compounds = [x["smi1"], x["smi2"]]
            compounds.sort()
            return {"_mixture_id": tuple(compounds)}

        ds = ds.map(label_mixture, batched=False)
        splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        ds = apply_splitter(ds, splitter, groups="_mixture_id", split=split_idx)

    else:
        raise ValueError(f"Unknown split type {split}")

    ds.save_to_disk(output)


if __name__ == "__main__":
    cli()
