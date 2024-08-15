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
from .molnet_dataset import MolNetDataModule


class PropertyPredictionDataModule(MolNetDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        smi_column: str = "smiles",
        target_columns: list[str] = ["Class"],
        strip_unk_tokens: bool = False,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir()

        self.smi_column = smi_column
        self.target_columns = target_columns
        self.strip_unk_tokens = strip_unk_tokens

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

        # # Inject methods from MolNetDataModule
        # self.data_collator = MolNetDataModule.data_collator
        # self.setup = MolNetDataModule.setup
        #

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        # Load datasets, checking for splits
        ds = load_dataset(str(self.path), keep_in_memory=False, streaming=True)
        assert "train" in ds and "validation" in ds
        self.__dataset = ds
        return self.__dataset
