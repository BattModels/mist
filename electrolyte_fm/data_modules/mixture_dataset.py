from pathlib import Path
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
import typer
from lightning import LightningDataModule
from datasets import Dataset, IterableDatasetDict, load_dataset
from torch.utils.data import default_collate, DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ..utils.tokenizer import load_tokenizer
from .utils import MolEncoding, stack_columns, collate_target
from .property_prediction_dataset import PropertyPredictionDataModule

cli = typer.Typer()


def collate_components_and_environment(
    *args,
    include_temperature,
    tokenizer,
    randomize,
    encoding,
    n_components,
    encoder: PreTrainedModel = None,
    collate: DataCollatorWithPadding = None,
):
    output = {}
    if include_temperature:
        temperature = torch.tensor(args[-1])
        output = {"temperature": temperature}
    for i in range(n_components):
        idx = 2 * i
        smiles = args[idx]
        if randomize:
            smiles = encoding.random(smiles)
        composition = args[idx + 1]
        batch = tokenizer(smiles)
        batch = collate(batch)
        output[f"input_ids_{i}"] = batch["input_ids"]
        output[f"attention_mask_{i}"] = batch["attention_mask"]
        output[f"composition_{i}"] = composition

    return output


def collate_target_stacked(x, target_columns):
    """Stack multiple target columns into a single vector,
    recording unknown elements to be masked out during training
    """
    target = []
    mask = []
    for k in target_columns:
        v = x[k]
        if v is None:
            target.append(torch.tensor(0))  # Placeholder, should be masked out
            mask.append(torch.tensor(False))
        else:
            target.append(torch.tensor(v))
            mask.append(torch.tensor(True))

    return {"target": torch.stack(target), "target_mask": torch.stack(mask)}


class ComponentDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        target_col: list | str,
        n_components: int = 2,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        include_temperature: str | bool = False,
        encoding: str | MolEncoding = "smiles",
        iterable: bool = False,
        randomize: bool = False,
        max_length: int = 512,
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.tokenizer = load_tokenizer(tokenizer)
        self.iterable = iterable
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        self.encoding = MolEncoding(encoding)
        self.randomize = randomize
        if isinstance(target_col, str):
            target_col = [
                target_col,
            ]
        self.target_col = target_col
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.n_components = n_components
        self.temperature = bool(include_temperature)
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.save_hyperparameters(logger=False)
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, max_length=max_length, padding="max_length"
        )

    def prepare_data(self):
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        if self.iterable:
            self._dataset = load_dataset(
                "arrow",
                name=str(self.path.name),
                data_files={
                    "train": str(self.path.joinpath("train/*.arrow")),
                    "validation": str(self.path.joinpath("validation/*.arrow")),
                    "test": str(self.path.joinpath("test/*.arrow")),
                },
                keep_in_memory=False,
                streaming=True,
                save_infos=False,
            )
            assert isinstance(self._dataset, IterableDatasetDict)
        else:
            self._dataset = self._dataset = load_dataset(
                "csv",
                name=str(self.path.name),
                data_files={
                    "train": str(self.path.joinpath("train.csv")),
                    "validation": str(self.path.joinpath("val.csv")),
                    "test": str(self.path.joinpath("test.csv")),
                },
            )
        return self._dataset

    def setup(self, stage: str) -> None:
        # Extract per molecule hidden states
        input_columns = []
        for i in range(self.n_components):
            input_columns.extend([f"smi{i+1}", f"x{i+1}"])
        if self.temperature:
            input_columns.append("temperature")

        ds = self.dataset
        ds = ds.map(
            collate_target_stacked,
            batched=False,
            fn_kwargs={"target_columns": self.target_col},
            remove_columns=self.target_col,
        )
        ds = ds.map(
            collate_components_and_environment,
            batched=False,
            fn_kwargs={
                "include_temperature": self.temperature,
                "tokenizer": self.tokenizer,
                "randomize": self.randomize,
                "encoding": self.encoding,
                "collate": self.data_collator,
                "n_components": self.n_components,
            },
            input_columns=input_columns,
        )
        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        self.target_dataset = (
            ds["train"].take(10000).select_columns(["target", "target_mask"])
        )

    def collator(self, batch):
        output = {}

        for i in range(self.n_components):
            output[f"input_ids_{i}"] = torch.stack(
                [torch.tensor(x[f"input_ids_{i}"], dtype=int) for x in batch]
            )
            output[f"attention_mask_{i}"] = torch.stack(
                [torch.tensor(x[f"attention_mask_{i}"], dtype=int) for x in batch]
            )
            output[f"composition_{i}"] = torch.stack(
                [
                    torch.tensor(x[f"composition_{i}"], dtype=torch.float32)
                    for x in batch
                ]
            )

        output["target"] = torch.stack(
            [torch.tensor(x["target"], dtype=torch.float32) for x in batch]
        )
        output["target_mask"] = torch.stack(
            [torch.tensor(x["target_mask"], dtype=bool) for x in batch]
        )

        if self.temperature:
            output["temperature"] = torch.stack(
                [torch.tensor(x["temperature"], dtype=torch.float32) for x in batch]
            )
        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
            persistent_workers=self.num_workers > 0,
        )


class ComponentDataModuleFast(PropertyPredictionDataModule):
    def __init__(
        self,
        path: str | Path,
        n_components: int = 2,
        temperature_column: Optional[str] = "temperature [kelvin]",
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
            fn_kwargs={"target_columns": self.target_columns, "dtype": torch.float32},
            remove_columns=self.target_columns,
        )

        if self.excess_columns:
            ds = ds.map(
                collate_target,
                batched=False,
                fn_kwargs={
                    "target_columns": self.excess_columns,
                    "name": "target_excess",
                    "dtype": torch.float32,
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

    @torch.no_grad()
    def collate_fn(self, batch):
        out = {
            k: default_collate([obs[k] for obs in batch])
            for k in batch[0].keys()
            if k not in ["compounds"]
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


def stack_compounds(row: dict, smi_columns: List[str]):
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
    compounds: List[Tuple[str, ...]],
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
    num_shards: int = 4,
    target_columns: Optional[List[str]] = None,
):
    import logging
    from .molnet_dataset import train_val_test_split
    from .splits import (
        EntityHoldoutSplitter,
        stratified_mixture_sparsity_split,
    )

    logging.basicConfig(level=logging.INFO)

    ds: Dataset = load_dataset("csv", data_files=[str(data)], split="train")
    ds = ds.map(lambda x: {"x2": 1 - x["x1"]}, batched=False)
    ds = ds.filter(
        lambda x: abs(x["pressure [megapascal]"] - 0.1) <= 0.02, batched=False
    )

    if split == "random":
        ds = train_val_test_split(ds)
        ds.save_to_disk(output, num_shards={k: num_shards for k in ds.keys()})
        return

    if split == "k-compound":
        splitter = EntityHoldoutSplitter(entity_cols=["inchi1", "inchi2"], verbose=True)
    elif split == "k-compound-strict":
        splitter = EntityHoldoutSplitter(
            entity_cols=["inchi1", "inchi2"], strict=True, verbose=True
        )
    else:
        raise ValueError(f"Unknown split type {split}")

    target_columns = target_columns or ["density", "molar volume", "molar enthalpy"]
    for idx, ds in enumerate(
        stratified_mixture_sparsity_split(ds, splitter, target_columns)
    ):
        name = Path(output.parent, output.name + f"-{idx}")
        ds.save_to_disk(name, num_shards={k: num_shards for k in ds.keys()})
        logging.debug("Saved split %d to %s", idx, name)


if __name__ == "__main__":
    cli()
