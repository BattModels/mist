import os
from pathlib import Path

import torch
from typing import Optional
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name):
        df = df[['smiles', measure_name]]
        df = df.dropna()
        self.measure_name = measure_name
        self.df = df.reset_index(drop=True)
        self._epoch = 0

    def __getitem__(self, index):
        return self.df.loc[index, 'smiles'], self.df.loc[index, self.measure_name]
  
    def __len__(self):
        return len(self.df)
    
    def set_epoch(self, epoch: int):
        self._epoch = epoch

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            path: str,
            tokenizer: str,
            dataset_name: str,
            measure_name: str,
            batch_size: int = 64,
            num_workers: int = 1,
            prefetch_factor: int = 4,
            val_batch_size: Optional[int] = None,
            train_dataset_length: Optional[int] = None,
            val_dataset_length: Optional[int] = None,
            test_dataset_length: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
        )
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.path: Path = Path(path)
        assert self.path.is_dir() or self.path.is_file()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset_name = dataset_name
        self.train_dataset_length = train_dataset_length
        self.val_dataset_length = val_dataset_length
        self.test_dataset_length = test_dataset_length
        self.measure_name = measure_name
        self.save_hyperparameters()

    def _get_split_dataset_filename(self, dataset_name, split):
        return os.path.join(dataset_name, split + ".csv")

    def setup(self, stage: str) -> None:

        train_filename = self._get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = self._get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = self._get_split_dataset_filename(
            self.dataset_name, "test"
        )

        self.train_dataset = get_dataset(
            self.path,
            train_filename,
            self.train_dataset_length,
            measure_name=self.measure_name,
        )

        self.val_dataset = get_dataset(
            self.path,
            valid_filename,
            self.val_dataset_length,
            measure_name=self.measure_name,
        )

        self.test_dataset = get_dataset(
            self.path,
            test_filename,
            self.test_dataset_length,
            measure_name=self.measure_name,
        )

    def data_collator(self, batch):
        tokens = self.tokenizer.batch_encode_plus(
            [smile[0] for smile in batch], 
            padding=True, 
            return_tensors="pt",
            add_special_tokens=True
            )
        tokens["targets"] = torch.tensor([smile[1] for smile in batch])
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
            shuffle=False
        )

    def test_dataset(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
    
def get_dataset(data_root, filename, dataset_len, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df,  measure_name)
    return dataset