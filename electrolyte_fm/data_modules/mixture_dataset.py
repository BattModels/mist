from pathlib import Path
from itertools import chain

import torch
import typer
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from torch.utils.data import default_collate

from .utils import MolEncoding, stack_columns
from .property_prediction_dataset import PropertyPredictionDataModule, collate_target

cli = typer.Typer()


class ComponentDataModule(PropertyPredictionDataModule):
    def __init__(
        self,
        path: str | Path,
        n_components: int = 2,
        temperature_column: str | None = "temperature [kelvin]",
        smi_column: str = "smi{:d}",
        x_column: str = "x{:d}",
        excess_columns: str | list[str] | None = "excess {:s}",
        **kwargs,
    ):
        self.path = Path(path)
        self.temperature_column = temperature_column
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
            fn_kwargs={
                "columns": self.x_columns,
                "output": "composition",
                "dtype": torch.float32,
            },
            remove_columns=self.x_columns,
        )

        # Stack compounds
        ds = ds.map(
            stack_compounds,
            batched=False,
            fn_kwargs={"smi_columns": self.smi_columns},
            remove_columns=self.smi_columns,
        )

        # Columns in final dataset
        columns = ["target", "target_mask", "composition", "compounds"]

        # Normalize temperature column
        if self.temperature_column is not None:
            if self.temperature_column != "temperature":
                ds = ds.rename_column(self.temperature_column, "temperature")
            columns.append("temperature")
            ds = ds.map(
                cast_dtype,
                batched=True,
                fn_kwargs={"column": "temperature", "dtype": torch.float32},
            )

        # Filter to input columns
        if self.excess_columns:
            columns.extend(["target_excess", "target_excess_mask"])

        ds = ds.filter(
            has_training_data,
            batched=False,
            fn_kwargs={"has_excess": self.excess_columns is not None},
        )

        ds = ds.select_columns(columns)

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        if "test" in ds:
            self.test_dataset: Dataset = ds["test"]

        # Dataset for normalization
        norm_columns = {
            "target",
            "target_mask",
            "temperature",
            "target_excess",
            "target_excess_mask",
        }
        norm_columns = list(set(norm_columns).intersection(columns))
        self.target_dataset = ds["train"].select_columns(norm_columns)

    def collate_fn(self, batch):
        out = {
            k: default_collate([obs[k] for obs in batch])
            for k in batch[0].keys()
            if k != "compounds"
        }
        out.update(
            encode_and_tokenize_mixture(
                [obs["compounds"] for obs in batch],
                tokenizer=self.tokenize,
                encoding=self.encoding,
                randomize=self.randomize,
                token_collator=self.token_collator,
                include_encoding=self.include_encoding,
            )
        )

        return out


def stack_compounds(row: dict, smi_columns: list[str]):
    return {"compounds": tuple(row[col] for col in smi_columns)}


def cast_dtype(row: dict, column: str, dtype: torch.dtype):
    return {column: torch.tensor(row[column], dtype=dtype)}


def has_training_data(x, has_excess: bool = True) -> bool:
    n_targets = x["target_mask"].sum() > 0
    if has_excess:
        n_excess = x["target_excess_mask"].sum() > 0
        return n_targets or n_excess
    return n_targets


def encode_and_tokenize_mixture(
    compounds: list[tuple[str, ...]],
    tokenizer=None,
    encoding: MolEncoding = MolEncoding.SMILES,
    randomize: bool = True,
    token_collator=None,
    include_encoding: bool = False,
):
    # Encode molecules
    encode = encoding.random if randomize else encoding
    smis = chain(*compounds)
    smis = [encode(smi) if smi else "" for smi in smis]
    toks = token_collator(tokenizer(smis))

    # Collate
    out = {}
    B = len(compounds)
    N = len(compounds[0])
    for k in ["input_ids", "attention_mask"]:
        out[k] = torch.reshape(toks[k], (B, N, -1))

    if include_encoding:
        out["compounds"] = compounds

    return out


@cli.command()
def split_dataset(
    data: Path,
    output: Path,
    split: str = "random",
    split_idx: int = 0,
    test: bool = False,
):
    from .molnet_dataset import train_val_test_split
    from .splits import (
        StrictEntityHoldoutSplitter,
        EntityHoldoutSplitter,
        apply_splitter,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from datasets import DatasetDict, concatenate_datasets

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

    if not test:
        ds = DatasetDict(
            {
                "train": ds["train"],
                "validation": concatenate_datasets(
                    [ds[k] for k in ["validation", "test"]]
                ),
            }
        )
    ds.save_to_disk(output)


if __name__ == "__main__":
    cli()
