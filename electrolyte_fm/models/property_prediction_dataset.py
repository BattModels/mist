import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


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
    def __init__(
            self, 
            path: str,
            tokenizer: str,
            dataset_name: str,
            batch_size: int = 64,
            val_batch_size=None,
            num_workers=1,
            prefetch_factor=4,
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
        self.save_hyperparameters()

    def get_split_dataset_filename(dataset_name, split):
        return dataset_name + "_" + split + ".csv"

    def prepare_data(self):

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

    def data_collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus(
            [smile[0] for smile in batch], 
            padding=True, 
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