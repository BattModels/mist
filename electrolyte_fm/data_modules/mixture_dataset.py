from pathlib import Path
from typing import List, Optional, Union

import torch
from datasets import Dataset, IterableDatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ..utils.tokenizer import load_tokenizer
from .utils import MolEncoding


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


def collate_target(x, target_columns):
    """Stack multiple target columns into a single vector,
    recording unknown elements to be masked out during training
    """
    target = []
    mask = []
    for k in target_columns:
        v = x[k]
        if v is None:
            target.append(torch.tensor(0))  # Placeholder, should be masked out
            mask.append(torch.tensor(True))
        else:
            target.append(torch.tensor(v))
            mask.append(torch.tensor(False))

    return {"target": torch.stack(target), "target_mask": torch.stack(mask)}


class ComponentDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        target_col: Union[str, List],
        n_components: int = 2,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        include_temperature: str | bool = False,
        encoding: Optional[str | MolEncoding] = "smiles",
        encoder_device: str = "cuda",
        iterable: bool = False,
        randomize: bool = False,
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
        self.encoder_device = torch.device(encoder_device)
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.n_components = n_components
        self.temperature = bool(include_temperature)
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.save_hyperparameters(logger=False)
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, max_length=128, padding="max_length"
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
            collate_target,
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
        self.target_dataset = ds["train"].select_columns(["target", "target_mask"])

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
