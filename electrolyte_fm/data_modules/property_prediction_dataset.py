import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from ..utils.tokenizer import load_tokenizer
from .roberta_dataset import maybe_shard_dataset


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        smi_column: str = "smiles",
        target_columns: list[str] = ["Class"],
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir()

        self.smi_column = smi_column
        self.target_columns = target_columns

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        # Load datasets, checking for splits
        ds = load_dataset(str(self.path), keep_in_memory=False, streaming=True)
        assert "train" in ds and "validation" in ds

        # Remove extraneous columns and tokenize smiles
        ds = ds.select_columns([self.smi_column, *self.target_columns])
        ds = ds.map(
            lambda x: self.tokenizer(x[self.smi_column]),
            batched=True,
            remove_columns=self.smi_column,
        )

        # Stack multiple target columns into a single vector, recording unknown elements
        # to be masked out during training
        def collate_target(x):
            target = []
            mask = []
            for k in self.target_columns:
                v = x[k]
                if v is None:
                    target.append(torch.tensor(0))  # Placeholder, should be masked out
                    mask.append(torch.tensor(0))
                else:
                    target.append(torch.tensor(v))
                    mask.append(torch.tensor(1))

            return {"target": torch.stack(target), "target_mask": torch.stack(mask)}

        ds = ds.map(collate_target, batched=False, remove_columns=self.target_columns)

        self.train_dataset: Dataset = maybe_shard_dataset(
            self.trainer, ds["train"].shuffle(seed=42)
        )
        self.val_dataset: Dataset = maybe_shard_dataset(self.trainer, ds["validation"])
        self.test_dataset: Dataset = maybe_shard_dataset(self.trainer, ds["test"])
        self.token_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding="longest"
        )

    def data_collator(self, batch):
        targets = [x.pop("target") for x in batch]
        mask = [x.pop("target_mask") for x in batch]
        output = self.token_collator(batch)
        output["target"] = torch.stack(targets)
        output["target_mask"] = torch.stack(mask)
        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
