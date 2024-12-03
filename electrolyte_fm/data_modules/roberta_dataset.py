from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from datasets import IterableDataset, DatasetDict, IterableDatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from ..utils.tokenizer import load_tokenizer
from .utils import MolEncoding, encode_molecules


def maybe_shard_dataset(trainer, ds):
    """Maybe shard a dataset across trainer ranks, if appropriate"""
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        return ds.__class__({k: maybe_shard_dataset(trainer, v) for k, v in ds.items()})
    if trainer is None:
        return ds
    return split_dataset_by_node(ds, trainer.global_rank, trainer.world_size)


class RobertaDataSet(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        mlm_probability=0.15,
        batch_size: int = 64,
        val_batch_size=None,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        canonical: Optional[bool] = None,  # Deprecated: Use encoding instead
        encoding: str | MolEncoding = "smiles",
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir() or self.path.is_file()

        # Handle canonical
        if canonical is not None:
            print("WARNING: canonical is deprecated, use encoding instead")
            if MolEncoding(encoding) == MolEncoding.SMILES:
                encoding = MolEncoding.CANONICAL_SMILES
            else:
                raise ValueError(f"Canonical encoding not supported for {encoding}")

        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.encoding = MolEncoding(encoding)
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        self._dataset = load_dataset(
            "text",
            name=str(self.path.name),
            data_files={
                "train": str(self.path.joinpath("data/train/*.txt")),
                "validation": str(self.path.joinpath("data/val/*.txt")),
                "test": str(self.path.joinpath("data/test/*.txt")),
            },
            keep_in_memory=False,
            streaming=True,
            save_infos=True,
        )
        return self._dataset

    def setup(self, stage: str) -> None:
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            mlm=True,
        )
        ds = maybe_shard_dataset(self.trainer, self.dataset)
        ds = ds.shuffle(
            seed=42,
            buffer_size=100 * max(self.val_batch_size, self.batch_size),
        )

        # Transcode
        ds = encode_molecules(ds, "text", encoding=self.encoding)

        # Tokenize
        ds = ds.map(
            self.tokenizer,
            batched=True,
            input_columns="text",
            remove_columns="text",
        )

        # Partition Datasets
        self.train_dataset: IterableDataset = ds["train"]
        self.val_dataset: IterableDataset = ds["validation"]
        self.test_dataset: IterableDataset = ds["test"]

    def train_dataloader(self):
        # Increment epoch to replicate shuffling
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def state_dict(self):
        state = {
            "path": str(self.path),
            "vocab_size": self.vocab_size,
        }
        if self.trainer is not None:
            state["rank"] = self.trainer.global_rank
            state["world_size"] = self.trainer.world_size

        for dl in ["train_dataset", "val_dataset", "test_dataset"]:
            if hasattr(self, dl):
                state[dl] = getattr(self, dl).state_dict()

        return state

    def load_state_dict(self, state: dict):
        assert self.path == Path(state["path"])
        assert self.vocab_size == state["vocab_size"]
        for dl in ["train_dataset", "val_dataset", "test_dataset"]:
            if dl in state and hasattr(self, dl):
                getattr(self, dl).load_state_dict(state[dl])
