import torch
import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import normalize_smiles
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df,  measure_name, aug=True):
        df = df.dropna()  # TODO - Check why some rows are na
        self.df = df
        all_smiles = df["smiles"].tolist()
        self.original_smiles = []
        self.original_canonical_map = {
            smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
        }
        if measure_name:
            all_measures = df[measure_name].tolist()
            self.measure_map = {all_smiles[i]: all_measures[i] for i in range(len(all_smiles))}

        # Get the canonical smiles
        # Convert the keys to canonical smiles if not already

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")

        self.aug = aug
        self.is_measure_available = "measure" in df.columns

    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        canonical_smiles = self.original_canonical_map[original_smiles]
        return canonical_smiles, self.measure_map[original_smiles]

    def __len__(self):
        return len(self.original_smiles)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, dataset_name):
        super().__init__()
        # if type(hparams) is dict:
        #     hparams = Namespace(**hparams)
        # self.hparams = hparams
        #self.smiles_emb_size = hparams.n_embd
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

        train_ds = get_dataset(
            self.hparams.data_root,
            train_filename,
            self.hparams.train_dataset_length,
            self.hparams.aug,
            measure_name=self.hparams.measure_name,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        self.train_ds = train_ds
        self.val_ds = [val_ds] + [test_ds]

        # print(
        #     f"Train dataset size: {len(self.train_ds)}, val: {len(self.val_ds1), len(self.val_ds2)}, test: {len(self.test_ds)}"
        # )

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
    
def get_dataset(data_root, filename, dataset_len, aug, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df,  measure_name, aug)
    return dataset