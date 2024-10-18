import re
from math import floor
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ..models.model_utils import load_encoder
from ..utils.tokenizer import load_tokenizer
from .roberta_dataset import maybe_shard_dataset


def extract_hidden_state(
    input_ids,
    attention_mask,
    encoder: PreTrainedModel = None,
    collate: DataCollatorWithPadding = None,
    layer: Union[int, float] = 0.5,
    device="cpu",
):
    batch = collate({"input_ids": input_ids, "attention_mask": attention_mask})
    attention_mask = batch["attention_mask"]
    batch = batch.to(encoder.device)

    # Disable gradients
    with torch.inference_mode():
        enc = encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )

    if isinstance(layer, float):
        layer = floor(len(enc["hidden_states"]) * layer)

    # Flatten hidden states
    hs = enc["hidden_states"][layer].to(device)
    hidden_state = []
    d_model = hs.shape[-1]
    assert attention_mask.shape[0] == hs.shape[0], "batch size mismatch"
    assert attention_mask.shape[1] == hs.shape[1], "seq. length mismatch"
    for bdx in range(hs.shape[0]):
        hs_molecule = hs[bdx][attention_mask[bdx] > 0]
        assert hs_molecule.shape == (len(input_ids[bdx]), d_model)
        hidden_state.append(hs_molecule)

    return {"hidden_state": hidden_state}




class HiddenStateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name_or_path: str,
        path: str,
        layer: Union[int, float] = 0.5,
        tokenizer: Optional[str] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        encoder_batch_size: Optional[int] = None,
        encoder_device: str = "cuda",
        return_molecule: bool = False,
        canonical=False,
    ):
        super().__init__()

        # Locate Tokeniser and dataset
        self.name_or_path = name_or_path
        self.layer = layer
        self.tokenizer = load_tokenizer(tokenizer or name_or_path)
        self.encoder_device = torch.device(encoder_device)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        self.return_molecule = return_molecule
        assert self.path.is_dir() or self.path.is_file()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.encoder_batch_size = encoder_batch_size or batch_size
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
        self.encoder = load_encoder(self.name_or_path).to(self.encoder_device)
        ds = maybe_shard_dataset(self.trainer, self.dataset)
        ds = ds.map(
            self.tokenizer,
            batched=True,
            input_columns="text",
            remove_columns="text",
        )
        tok_columns = ["input_ids", "attention_mask", ]
        ds = ds.select_columns(tok_columns)

        # Extract per molecule hidden states
        ds = ds.map(
            extract_hidden_state,
            batched=True,
            batch_size=self.encoder_batch_size,
            fn_kwargs={
                "encoder": self.encoder,
                "layer": self.layer,
                "collate": self.data_collator,
            },
            input_columns=tok_columns,
        )
        ds = ds.map(
            lambda x: {"hidden_state": torch.cat(x, dim=0)},
            batched=True,
            input_columns=["hidden_state"],
            remove_columns=tok_columns,
        )

        self.train_dataset: Dataset = ds["train"].shuffle(
            buffer_size=10 * self.encoder_batch_size
        )
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
        )

    @classmethod
    def collate_fn(cls, batch):
        return torch.stack([x["hidden_state"] for x in batch]).detach()

class FeatureTaggerDataModule(HiddenStateDataModule):
    def __init__(self, name_or_path: str, path: str, features: dict[str, re.Pattern], **kwargs):
        kwargs["return_molecule"] = True
        self.features = features
        super().__init__(name_or_path, path, **kwargs)

    def setup(self, stage: str) -> None:
        self.encoder = load_encoder(self.name_or_path).to(self.encoder_device)
        ds = maybe_shard_dataset(self.trainer, self.dataset)
        ds = ds.map(
            self.tokenizer,
            batched=True,
            input_columns="text",
            fn_kwargs={"return_offsets_mapping": True},
        )
        tok_columns = ["input_ids", "attention_mask", "offsets_mapping"]
        ds = ds.select_columns(tok_columns)

        # Extract per molecule hidden states
        ds = ds.map(
            extract_hidden_state,
            batched=True,
            batch_size=self.encoder_batch_size,
            fn_kwargs={
                "encoder": self.encoder,
                "layer": self.layer,
                "collate": self.data_collator,
            },
            input_columns=tok_columns,
        )

        ds = ds.map(
            tag_features,
            batched=False,
            input_columns=tok_columns,
            fn_kwargs={
                "features": self.features,
            },
        )

        # Tag features
        self.train_dataset: Dataset = ds["train"].shuffle(
            buffer_size=10 * self.encoder_batch_size
        )
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        super().setup(stage)
        self.train_dataset


def tag_features(obs:dict, features:dict):
    token_overlap = torch.zeros(len(features), len(obs["input_ids"]), dtype=torch.bool)
    offsets_mapping = obs["offsets_mapping"]
    for fdx, (name, pattern) in features.items():
        m = pattern.search(obs["text"])
        if m is None:
            continue
        for idx, (start, end) in enumerate(offsets_mapping):
            if start <= m.span[0] and end >= m.span[1]:
                token_overlap[idx, fdx] = True

    return {"features": token_overlap.T.detach()}







         
