import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union
import yaml
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from ..utils.tokenizer import load_tokenizer

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        task_config: str = "",
        dataset_name: str = "bace",
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        val_batch_size: Optional[int] = None,
        train_dataset_length: Optional[int] = None,
        val_dataset_length: Optional[int] = None,
        test_dataset_length: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir() or self.path.is_file()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset_name = dataset_name
        self.task_config: Path = Path(task_config) 
        assert self.task_config.is_file()
        self.setup_task_config()
        self.train_dataset_length = train_dataset_length
        self.val_dataset_length = val_dataset_length
        self.test_dataset_length = test_dataset_length
        self.save_hyperparameters(logger=False)

    def setup_task_config(self) -> None:
        self.task_specs = yaml.safe_load(self.task_config.open())

    def setup(self, stage: str) -> None:
        ds = load_dataset(os.path.join(self.path, self.dataset_name))

        # Setup to partition datasets over ranks
        if self.trainer is None:
            rank = 0
            world_size = 1
        else:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size
        ds_train: Dataset = ds["train"].shuffle(seed=42)

        self.train_dataset: Dataset = split_dataset_by_node(
            ds_train,
            rank=rank,
            world_size=world_size,
        )
        self.val_dataset: Dataset = split_dataset_by_node(
            ds["validation"],
            rank=rank,
            world_size=world_size,
        )
        self.test_dataset: Dataset = split_dataset_by_node(
            ds["test"],
            rank=rank,
            world_size=world_size,
        )
        self.calculate_imputation_values()

    def calculate_imputation_values(self):
        for spec in self.task_specs:
            if spec.get("n_classes", 1) > 1:
                spec["fill_value"] = -1
            else:
                spec["fill_value"] = mean(
                    d for d in self.train_dataset[spec["measure_name"]] if d is not None
                )

    def data_collator(self, batch):
        
        def _impute(val):
            if val is None:
                return spec["fill_value"]
            else:
                return val
            
        tokens = self.tokenizer._batch_encode_plus(
            [sample["smiles"] for sample in batch],
            add_special_tokens=True,
            return_tensors="pt",
            padding_strategy="longest",
        )
        for spec in self.task_specs:
            tokens[spec["measure_name"]] = torch.tensor(
                [_impute(sample[spec["measure_name"]]) for sample in batch]
            )
        return tokens

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
            shuffle=False,
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
