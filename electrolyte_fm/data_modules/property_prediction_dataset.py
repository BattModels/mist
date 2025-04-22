from abc import abstractmethod
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from datasets import Dataset
from transformers import DataCollatorWithPadding

from ..utils.tokenizer import load_tokenizer
from .utils import (
    MolEncoding,
    AbstractDataset,
    maybe_shard_dataset,
    encode_molecules,
    is_fast,
)


class PropertyPredictionDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: str = "smirk",
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        smi_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        encoding: str = MolEncoding.SMILES.value,
        additonal_columns: Optional[List[str]] = None,
        include_encoding: bool = False,
        randomize: bool = False,
    ):
        super().__init__()

        self.tokenizer = (
            load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.token_collator = DataCollatorWithPadding(self.tokenizer)
        self.vocab_size = len(self.tokenizer)

        self.smi_column = smi_column
        self.target_columns = target_columns
        self.additonal_columns = additonal_columns or []
        self.encoding = MolEncoding(encoding)
        self.include_encoding = include_encoding
        self.randomize = randomize

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor or (4 if num_workers > 0 else None)
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        self.__dataset = self._get_dataset()
        return self.__dataset

    @abstractmethod
    def _get_dataset(self) -> AbstractDataset:
        raise NotImplementedError()

    def setup(self, stage: str) -> None:
        # Load datasets, checking for splits
        ds = self.dataset
        ds = maybe_shard_dataset(self.trainer, ds)
        ds = encode_molecules(ds, self.smi_column, encoding=self.encoding)
        print(ds["train"])

        # Remove extraneous columns and tokenize smiles
        if targets := self.target_columns:
            ds = ds.map(
                collate_target,
                batched=False,
                fn_kwargs={"target_columns": targets},
                remove_columns=targets,
            )

            # Save training dataset for target transformations
            self.target_dataset = ds["train"].select_columns(["target", "target_mask"])
            ds = ds.select_columns(
                [self.smi_column, "target", "target_mask", *self.additonal_columns]
            )
        else:
            ds = ds.select_columns([self.smi_column, *self.additonal_columns])

        # Tokenize
        ds = ds.map(
            self.tokenizer,
            batched=is_fast(self.tokenizer),
            input_columns=self.smi_column,
        )
        if not self.include_encoding and not self.randomize:
            ds = ds.remove_columns(self.smi_column)

        self.train_dataset: Dataset = ds["train"].shuffle(seed=42)
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        self.token_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")

    def collate_fn(self, batch):
        tokenizer = self.tokenizer
        encoding = self.encoding
        if self.randomize:
            for idx in range(len(batch)):
                new_smi = encoding.random(batch[idx][self.smi_column])
                if new_smi is not None:
                    batch[idx].update(tokenizer(new_smi))

                if not self.include_encoding:
                    batch[idx].pop(self.smi_column, None)

        output = self.token_collator(batch)
        if self.target_columns:
            output["target"] = torch.stack([torch.tensor(x["target"]) for x in batch])
            output["target_mask"] = torch.stack(
                [torch.tensor(x["target_mask"]) for x in batch]
            )
            assert output["target"].shape == output["target_mask"].shape

        return output

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
        print("has validation dataloader")
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

    return {"target": target, "target_mask": mask}
