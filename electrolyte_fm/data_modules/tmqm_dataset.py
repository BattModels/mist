import gzip
import re
from pathlib import Path
from typing import Iterable, Optional
import logging

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding

from ..utils.tokenizer import load_tokenizer
from .roberta_dataset import maybe_shard_dataset
from .molnet_dataset import collate_target
from .utils import MolEncoding, encode_molecules


class tmQMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        mlm_probability=0.15,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_columns: Optional[list[str]] = None,
        encoding: str | MolEncoding = "smiles",
        include_encoding: bool = False,
        smi_column: str = "smiles",
    ):
        super().__init__()

        self.path = Path(path)
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)

        self.target_columns = target_columns
        self.encoding = MolEncoding(encoding)
        self.smi_column = smi_column
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
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
            save_infos=True,
        )
        ds = maybe_shard_dataset(self.trainer, ds)

        if targets := self.target_columns:
            ds = ds.map(
                collate_target, batched=False, fn_kwargs={"target_columns": targets}
            )
            self.target_dataset = ds["train"].select_columns(["target", "target_mask"])

        # Transcode
        ds = encode_molecules(ds, self.smi_column, encoding=self.encoding)

        # Tokenize
        ds = ds.map(self.tokenizer, batched=True, input_columns=self.smi_column)
        self.train_dataset = ds["train"]
        self.val_dataset = ds["validation"]
        self.test_dataset = ds["test"]
        self.token_collator = DataCollatorWithPadding(self.tokenizer, "longest")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def collate_fn(self, batch):
        # Collate tokens
        token_inputs = ["input_ids", "attention_mask"]
        token_inputs = [{k: x[k] for k in token_inputs} for x in batch]
        output = self.token_collator(token_inputs)

        # Add targets and mask
        if "target" in batch[0].keys():
            output["target"] = torch.stack([x["target"].detach() for x in batch])

        if "target_mask" in batch[0].keys():
            output["target_mask"] = torch.stack(
                [x["target_mask"].detach() for x in batch]
            )

        return output
