import re
from math import floor
from pathlib import Path
from typing import Optional, Union

import torch
import nvtx
import pytorch_lightning as pl
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ..models.model_utils import load_encoder
from ..utils.tokenizer import load_tokenizer
from .utils import MolEncoding, encode_molecules


def extract_hidden_state(
    sub1_smiles,
    x1,
    sub2_smiles,
    x2,
    target,
    tokenizer,
    encoder: PreTrainedModel = None,
    collate: DataCollatorWithPadding = None,
):
    mix_embedding = None

    for smiles, composition in [(sub1_smiles, x1), (sub2_smiles, x2)]:
        batch = tokenizer(smiles)
        batch = collate(batch)
        batch = batch.to(encoder.device)
        # Disable gradients
        with nvtx.annotate("encoder"):
            with torch.inference_mode():
                embedding = (
                    encoder(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    .last_hidden_state.detach()
                    .cpu()
                    .mean(axis=1)
                )
                update = torch.stack(
                    [
                        torch.mul(embedding[i, :], composition[i])
                        for i in range(embedding.shape[0])
                    ]
                )
                if mix_embedding is None:
                    mix_embedding = update
                else:
                    mix_embedding += update

    return {"embedding": mix_embedding, "target": target}


class HiddenStateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name_or_path: str,
        path: str,
        target_col: str,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        encoder_batch_size: Optional[int] = None,
        encoder_device: str = "cuda",
        return_molecule: bool = False,
        encoding: Optional[str | MolEncoding] = "smiles",
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.name_or_path = name_or_path
        self.tokenizer = load_tokenizer(tokenizer or name_or_path)
        self.encoder_device = torch.device(encoder_device)
        self.path: Path = Path(path)
        self.return_molecule = return_molecule
        self.encoding = MolEncoding(encoding)
        self.target_col = target_col
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.encoder_batch_size = encoder_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["tokenizer"] = tokenizer
        self.save_hyperparameters(logger=False, ignore=["encoder_device"])
        self.data_collator = DataCollatorWithPadding(self.tokenizer, "longest")

    def prepare_data(self):
        self.dataset

    @property
    def dataset(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        self._dataset = load_dataset(
            "csv",
            name=str(self.path.name),
            data_files={
                "train": str(self.path.joinpath("train.csv")),
                "validation": str(self.path.joinpath("val.csv")),
                "test": str(self.path.joinpath("test.csv")),
            },
            keep_in_memory=True,
        )
        return self._dataset

    def setup(self, stage: str) -> None:
        self.encoder = load_encoder(self.name_or_path).to(self.encoder_device)

        # Extract per molecule hidden states
        input_columns = ["sub1_smiles", "x1", "sub2_smiles", "x2", self.target_col]
        ds = self.dataset
        ds = ds.map(
            extract_hidden_state,
            batched=True,
            batch_size=self.encoder_batch_size,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "encoder": self.encoder,
                "collate": self.data_collator,
            },
            input_columns=input_columns,
            remove_columns=input_columns,
        )

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]

        self.target_dataset = ds["train"].select_columns(["target"])

    def collator(self, batch):
        output = {}
        output["embedding"] = torch.stack([torch.tensor(x["embedding"]) for x in batch])
        output["target"] = torch.stack(
            [torch.tensor(x["target"], dtype=float) for x in batch]
        )
        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.val_batch_size,
        )
