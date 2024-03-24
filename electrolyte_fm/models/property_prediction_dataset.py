import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils import normalize_smiles


class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name):
        df = df[['smiles', measure_name]]
        df = df.dropna()
        self.measure_name = measure_name
        self.df = df.reset_index(drop=True)

    def __getitem__(self, index):
        return self.df.loc[index, 'smiles'], self.df.loc[index, self.measure_name]
  
    def __len__(self):
        return len(self.df)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, dataset_name):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
        )
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.dataset_name = dataset_name
        self.save_hyperparameters()

    def get_split_dataset_filename(dataset_name, split):
        return dataset_name + "_" + split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
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
            self.hparams.data_root,
            test_filename,
            self.val_dataset_length,
            measure_name=self.measure_name,
        )

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch], padding=True, add_special_tokens=True)
        tokens["targets"] = torch.tensor([smile[1] for smile in batch])
        return tokens
    
    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                collate_fn=self.collate,
            )
            for ds in self.val_ds
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )
    
def get_dataset(data_root, filename, dataset_len, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df,  measure_name)
    return dataset