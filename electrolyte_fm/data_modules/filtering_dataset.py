import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from lightning import LightningDataModule
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from ..utils.tokenizer import load_tokenizer
from .roberta_dataset import maybe_shard_dataset
from .utils import MolEncoding, encode_molecules
from .molnet_data import MolNetDataModule, scaffold_split, train_val_test_split


class FilteringDataset(MolNetDataModule):
    """
    Use QM9 regression datasets to do as thresholded classification.
    """

    def __init__(
        self,
        name: str = "qm9",
        split: str = "random",
        tokenizer: str = "smirk",
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        smi_column: Optional[str] = None,
        target_columns: list[str] = ["gap"],
        thresholds: list[float] = 0.5,
        val_batch_size: Optional[int] = None,
        encoding: Optional[str | MolEncoding] = None,
        include_encoding: bool = False,
    ):
        super().__init__()

        self.thresholds = thresholds
        assert len(self.thresholds) == len(self.target_columns)

    def setup(self, stage: str) -> None:
        # Load datasets, checking for splits
        ds = self.dataset
        ds = maybe_shard_dataset(self.trainer, ds)
        ds = encode_molecules(ds, self.smi_column, encoding=self.encoding)

        # Remove extraneous columns and tokenize smiles
        targets = self.target_columns
        ds = ds.map(
            collate_target,
            batched=False,
            fn_kwargs={"target_columns": targets, "thresholds": self.thresholds},
            remove_columns=targets,
        )

        # Save training dataset for target transformations
        self.target_dataset = ds["train"].select_columns(["target", "target_mask"])
        ds = ds.select_columns([self.smi_column, "target", "target_mask"])

        # Tokenize
        ds = ds.map(self.tokenizer, batched=True, input_columns=self.smi_column)
        if not self.include_encoding:
            ds = ds.remove_columns(self.smi_column)

        self.train_dataset: Dataset = ds["train"].shuffle(seed=42)
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        self.token_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")


def collate_target(x, target_columns, thresholds):
    """Stack multiple target columns into a single vector,
    recording unknown elements to be masked out during training
    """
    target = []
    mask = []
    for i in range(len(target_columns)):
        k = target_columns[i]
        v = x[k]
        if v is None:
            target.append(torch.tensor(0))  # Placeholder, should be masked out
            mask.append(torch.tensor(True))
        else:
            if v > thresholds[i]:
                target.append(torch.tensor(1))  # positive label
            else:
                target.append(torch.tensor(0))  # negative label
            mask.append(torch.tensor(False))

    return {"target": torch.stack(target), "target_mask": torch.stack(mask)}
