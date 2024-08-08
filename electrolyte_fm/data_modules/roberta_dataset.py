from pathlib import Path

import pytorch_lightning as pl
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from ..utils.tokenizer import load_tokenizer


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
        canonical=False,
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir() or self.path.is_file()

        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.canonical = canonical
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        self.__load_dataset()

    def __load_dataset(self):
        if not hasattr(self, "dataset"):
            dataset = load_dataset(
                str(self.path),
                keep_in_memory=False,
                streaming=True,
            )
            assert isinstance(dataset, IterableDatasetDict)
            self.dataset = dataset

        return self.dataset

    def setup(self, stage: str) -> None:
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            mlm=True,
        )
        ds = self.__load_dataset()
        if self.canonical:
            from ..utils.tokenizer import rdkit_canonical

            ds = ds.map(
                lambda x: {"text": rdkit_canonical(x["text"])},
                batched=False,
            ).filter(lambda x: x["text"] is not None)

        ds = ds.map(
            lambda batch: self.tokenizer(batch["text"]),
            batched=True,
            remove_columns="text",
        )

        # Setup to partition datasets over ranks
        if trainer := self.trainer:
            assert trainer is not None
            rank = trainer.global_rank
            world_size = trainer.world_size
            ds["train"]: IterableDataset = ds["train"].shuffle(seed=42)
            assert ds["train"].n_shards % world_size == 0
            assert ds["validation"].n_shards % world_size == 0
            assert ds["test"].n_shards % world_size == 0

            def build_ds(ds):
                return split_dataset_by_node(ds, rank, world_size)

        else:

            def build_ds(ds):
                return ds

        # Partition Datasets
        self.train_dataset: IterableDataset = build_ds(ds["train"])
        self.val_dataset: IterableDataset = build_ds(ds["validation"])
        self.test_dataset: IterableDataset = build_ds(ds["test"])

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
