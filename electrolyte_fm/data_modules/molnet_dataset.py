import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union
import logging

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupShuffleSplit
import pytorch_lightning as pl
import torch
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from ..utils.tokenizer import load_tokenizer
from .roberta_dataset import maybe_shard_dataset

_URLS = {
    "qm8": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
    "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
    "lipo": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "muv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "toxcast": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
    "sider": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
}


class MolNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str = "bace",
        split: str = "random",
        tokenizer: str = "smirk",
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        smi_column: Optional[str] = None,
        target_columns: list[str] = ["Class"],
        strip_unk_tokens: bool = False,
        val_batch_size: Optional[int] = None,
        include_smiles=False,
    ):
        super().__init__()

        self.name = name
        assert name in _URLS, f"Unknown MoleculeDataset {name}"
        self.include_smiles = include_smiles

        # Set smi_column
        self.use_selfies = "selfies" in tokenizer

        if smi_column is None and name == "bace":
            self.smi_column = "mol"
        else:
            self.smi_column = smi_column or "smiles"

        assert self.smi_column is not None
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.split = split

        self.target_columns = target_columns
        self.strip_unk_tokens = strip_unk_tokens

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        # Fetch data from the head node
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        # Load the dataset
        ds: Dataset = load_dataset(
            "csv",
            name=self.name,
            data_files=[_URLS[self.name]],
            split="train",
            keep_in_memory=False,
            save_infos=True,
        )  # type: ignore

        if self.name == "qm8":
            # Rename qm8 columns to remove duplicates and include source theory
            ds = ds.rename_columns(
                {
                    "E1-CC2": "E1-CC2-RI-CC2/def2TZVP",
                    "E2-CC2": "E2-CC2-RI-CC2/def2TZVP",
                    "f1-CC2": "f1-CC2-RI-CC2/def2TZVP",
                    "f2-CC2": "f2-CC2-RI-CC2/def2TZVP",
                    "E1-PBE0": "E1-PBE0-LR-TDPBE0/def2SVP",
                    "E2-PBE0": "E2-PBE0-LR-TDPBE0/def2SVP",
                    "f1-PBE0": "f1-PBE0-LR-TDPBE0/def2SVP",
                    "f2-PBE0": "f2-PBE0-LR-TDPBE0/def2SVP",
                    "E1-PBE0.1": "E1-PBE0-LR-TDPBE0/def2TZVP",
                    "E2-PBE0.1": "E2-PBE0-LR-TDPBE0/def2TZVP",
                    "f1-PBE0.1": "f1-PBE0-LR-TDPBE0/def2TZVP",
                    "f2-PBE0.1": "f2-PBE0-LR-TDPBE0/def2TZVP",
                    "E1-CAM": "E1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "E2-CAM": "E2-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "f1-CAM": "f1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "f2-CAM": "f2-CAM-LR-TDCAM-B3LYP/def2TZVP",
                }
            )

        # Spit into train/val/test
        if self.split == "scaffold":
            ds = scaffold_split(ds, self.smi_column)
        elif self.split == "random":
            ds = train_val_test_split(ds)
        else:
            raise ValueError(f"Unknown split {self.split}")

        # Cache the dataset
        self.__dataset = ds
        return self.__dataset

    def setup(self, stage: str) -> None:
        # Load datasets, checking for splits
        ds = self.dataset

        # Remove extraneous columns and tokenize smiles
        ds = ds.select_columns([self.smi_column, *self.target_columns])
        ds = ds.map(
            collate_target,
            batched=False,
            remove_columns=self.target_columns,
            fn_kwargs={"target_columns": self.target_columns},
        )

        # Save training dataset for target transformations
        self.target_dataset = ds["train"].select_columns(["target", "target_mask"])

        if self.use_selfies:
            from selfies import encoder

            def selfies_encoder(smi):
                try:
                    selfie = encoder(smi)
                except Exception:
                    selfie = None
                return selfie

            ds = ds.map(
                lambda smi: {"selfies": selfies_encoder(smi[self.smi_column])},
                batched=False,
                remove_columns=self.smi_column,
            ).filter(lambda x: x["selfies"] is not None)

            self.smi_column = "selfies"

        # Tokenize smiles
        ds = ds.map(
            self.tokenizer,
            batched=True,
            input_columns=self.smi_column,
            fn_kwargs={
                "return_offsets_mapping": self.tokenizer.is_fast and self.include_smiles
            },
        )

        # Optionally include a smiles column
        if self.include_smiles:
            ds.remove_columns(self.smi_column)
        else:
            ds.rename_column(self.smi_column, "smiles")

        self.train_dataset: Dataset = maybe_shard_dataset(
            self.trainer, ds["train"].shuffle(seed=42)
        )
        self.val_dataset: Dataset = maybe_shard_dataset(self.trainer, ds["validation"])
        self.test_dataset: Dataset = maybe_shard_dataset(self.trainer, ds["test"])
        self.token_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding="longest"
        )

    def data_collator(self, batch):
        targets = [torch.tensor(x.pop("target")) for x in batch]
        mask = [torch.tensor(x.pop("target_mask"), dtype=bool) for x in batch]

        # Remove unknown tokens
        unk_token_id = self.tokenizer.unk_token_id
        if self.strip_unk_tokens:
            batch = [strip_unk_tokens(obs, unk_token_id) for obs in batch]
            is_oov = [x.pop("is_oov") for x in batch]
        else:
            is_oov = [unk_token_id in obs["input_ids"] for obs in batch]

        # Tokenize
        output = self.token_collator(batch)
        output["target"] = torch.stack(targets)
        output["target_mask"] = torch.stack(mask)
        output["is_oov"] = torch.tensor(is_oov)
        return output

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
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
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

    return {"target": torch.stack(target), "target_mask": torch.stack(mask)}


def train_val_test_split(ds, **kwargs):
    ds_train_other = ds.train_test_split(test_size=0.2, seed=42, **kwargs)
    ds_val_test = ds_train_other["test"].train_test_split(
        test_size=0.5, seed=42, **kwargs
    )
    return DatasetDict(
        {
            "train": ds_train_other["train"],
            "validation": ds_val_test["train"],
            "test": ds_val_test["test"],
        }
    )


def scaffold_hash(smi: str) -> str:
    try:
        scaffold = MurckoScaffoldSmiles(smi)
    except ValueError:
        logging.warn("No scaffold for %s, using input smiles string", smi)
        scaffold = smi
    return scaffold


def scaffold_split(ds: Dataset, smi_column):
    # Hash scaffolds and then bin into groups, maintains the scaffold split
    # but reduces the compute
    ds = ds.map(
        lambda x: {"scaffold": scaffold_hash(x)},
        input_columns=smi_column,
        batched=False,
    ).to_pandas(batched=False)

    # Split
    train, other = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            ds.index, groups=ds["scaffold"].values
        )
    )
    ds_other = ds.iloc[other]
    val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(
            ds_other, groups=ds.iloc[other]["scaffold"]
        )
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(ds.iloc[train], preserve_index=False),
            "validation": Dataset.from_pandas(ds_other.iloc[val], preserve_index=False),
            "test": Dataset.from_pandas(ds_other.iloc[test], preserve_index=False),
        }
    )


def strip_unk_tokens(batch: dict, unk_token_id: int) -> dict:
    """Remove unknown tokens from input"""
    is_oov = [id == unk_token_id for id in batch["input_ids"]]
    out = {}
    for k, v in batch.items():
        assert len(v) == len(is_oov)
        out[k] = [x for x, oov in zip(v, is_oov) if not oov]
    out["is_oov"] = any(is_oov)
    return out
