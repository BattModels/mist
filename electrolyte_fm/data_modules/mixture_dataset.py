import re
from math import floor
from pathlib import Path
from typing import Optional, Union

import torch
import nvtx
import pytorch_lightning as pl
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ..models.model_utils import load_encoder
from ..utils.tokenizer import load_tokenizer
from .utils import MolEncoding, encode_molecules


def extract_hidden_state(
    *args,
    temperature,
    tokenizer,
    n_components,
    encoder: PreTrainedModel = None,
    collate: DataCollatorWithPadding = None,
):
    mix_embedding = None

    target = args[-1]

    if temperature:
        temperature = args[-2]

    for i in range(n_components):
        idx = 2 * i
        smiles = args[idx]
        composition = args[idx + 1]
        batch = tokenizer(smiles)
        batch = collate(batch)
        batch = batch.to(encoder.device)
        # Disable gradients
        with nvtx.annotate("encoder"):
            with torch.inference_mode():
                embedding = (
                    encoder(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    .last_hidden_state.detach()
                    .cpu()
                    .mean(axis=1)
                )
                update = torch.stack(
                    [
                        torch.mul(embedding[i, :], composition[i])
                        for i in range(embedding.shape[0])
                    ]
                )
                if mix_embedding is None:
                    mix_embedding = update
                else:
                    mix_embedding += update
    temperature = (torch.tensor(temperature) - 273) / (400 - 273)
    mix_embedding = torch.hstack((temperature.view(-1, 1), mix_embedding))
    return {"embedding": mix_embedding, "target": target}


class HiddenStateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name_or_path: str,
        path: str,
        target_col: str,
        n_components: int = 2,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        encoder_batch_size: Optional[int] = None,
        encoder_device: str = "cuda",
        return_molecule: bool = False,
        include_temperature: bool = False,
        encoding: Optional[str | MolEncoding] = "smiles",
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.name_or_path = name_or_path
        self.tokenizer = load_tokenizer(tokenizer or name_or_path)
        self.encoder_device = torch.device(encoder_device)
        self.path: Path = Path(path)
        self.return_molecule = return_molecule
        self.encoding = MolEncoding(encoding)
        self.target_col = target_col
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.n_components = n_components
        self.temperature = include_temperature
        self.val_batch_size = val_batch_size or batch_size
        self.encoder_batch_size = encoder_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["tokenizer"] = tokenizer
        self.save_hyperparameters(logger=False, ignore=["encoder_device"])
        self.data_collator = DataCollatorWithPadding(self.tokenizer, "longest")

    def prepare_data(self):
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        self._dataset = load_dataset(
            "csv",
            name=str(self.path.name),
            data_files={
                "train": str(self.path.joinpath("train.csv")),
                "validation": str(self.path.joinpath("val.csv")),
                "test": str(self.path.joinpath("test.csv")),
            },
            keep_in_memory=True,
        )
        return self._dataset

    def setup(self, stage: str) -> None:
        self.encoder = load_encoder(self.name_or_path).to(self.encoder_device)

        # Extract per molecule hidden states
        input_columns = []
        for i in range(self.n_components):
            input_columns.extend([f"smi{i+1}", f"x{i+1}"])
        if self.temperature:
            input_columns.append("temperature")
        input_columns.append(self.target_col)
        ds = self.dataset
        ds = ds.map(
            extract_hidden_state,
            batched=True,
            batch_size=self.encoder_batch_size,
            fn_kwargs={
                "temperature": self.temperature,
                "tokenizer": self.tokenizer,
                "encoder": self.encoder,
                "collate": self.data_collator,
                "n_components": self.n_components,
            },
            input_columns=input_columns,
            remove_columns=input_columns,
        )

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]

        self.target_dataset = ds["train"].select_columns(["target"])

    def collator(self, batch):
        output = {}
        output["embedding"] = torch.stack([torch.tensor(x["embedding"]) for x in batch])
        output["target"] = torch.stack(
            [torch.tensor(x["target"], dtype=float) for x in batch]
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
        )


def collate_components_and_environment(
    *args,
    include_temperature,
    tokenizer,
    n_components,
    encoder: PreTrainedModel = None,
    collate: DataCollatorWithPadding = None,
):
    target = args[-1]

    if include_temperature:
        temperature = args[-2]
        temperature = (torch.tensor(temperature) - 273) / (400 - 273)

    output = {
        "target": target,
        "temperature": temperature,
    }

    for i in range(n_components):
        idx = 2 * i
        smiles = args[idx]
        composition = args[idx + 1]
        batch = tokenizer(smiles)
        batch = collate(batch)
        output[f"input_ids_{i}"] = batch["input_ids"]
        output[f"attention_mask_{i}"] = batch["attention_mask"]
        output[f"composition_{i}"] = composition

    return output


class ComponentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        # name_or_path: str,
        target_col: str,
        n_components: int = 2,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        encoder_batch_size: Optional[int] = None,
        include_temperature: bool = True,
        encoding: Optional[str | MolEncoding] = "smiles",
        encoder_device: str = "cuda",
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.tokenizer = load_tokenizer(tokenizer)
        self.path: Path = Path(path)
        # self.name_or_path = name_or_path
        self.encoding = MolEncoding(encoding)
        self.target_col = target_col
        self.encoder_batch_size = encoder_batch_size
        self.encoder_device = torch.device(encoder_device)
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.n_components = n_components
        self.temperature = include_temperature
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        # self.hparams["tokenizer"] = tokenizer
        self.save_hyperparameters(logger=False)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, "longest")

    def prepare_data(self):
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        self._dataset = load_dataset(
            "csv",
            name=str(self.path.name),
            data_files={
                "train": str(self.path.joinpath("train.csv")),
                "validation": str(self.path.joinpath("val.csv")),
                "test": str(self.path.joinpath("test.csv")),
            },
            keep_in_memory=True,
        )
        return self._dataset

    def setup(self, stage: str) -> None:
        # Extract per molecule hidden states
        input_columns = []
        for i in range(self.n_components):
            input_columns.extend([f"smi{i+1}", f"x{i+1}"])
        if self.temperature:
            input_columns.append("temperature")
        input_columns.append(self.target_col)

        ds = self.dataset
        ds = ds.map(
            collate_components_and_environment,
            batched=True,
            batch_size=self.encoder_batch_size,
            fn_kwargs={
                "include_temperature": self.temperature,
                "tokenizer": self.tokenizer,
                "collate": self.data_collator,
                "n_components": self.n_components,
            },
            input_columns=input_columns,
            remove_columns=input_columns,
        )

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]

        self.target_dataset = ds["train"].select_columns(["target"])

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
                [torch.tensor(x[f"composition_{i}"], dtype=float) for x in batch]
            )

        output["target"] = torch.stack(
            [torch.tensor(x["target"], dtype=float) for x in batch]
        )
        output["temperature"] = torch.stack(
            [torch.tensor(x["temperature"], dtype=float) for x in batch]
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
        )
