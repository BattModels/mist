import json
import logging
from math import floor
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torchmetrics import MetricCollection

from ..data_modules.sae_dataset import extract_hidden_state
from ..data_modules.feature_tagger import FeatureCollection
from ..data_modules.utils import MolEncoding, encode_molecules
from ..models.model_utils import load_encoder
from ..models.sae import SAE
from .tokenizer import load_tokenizer
from .metrics import FeatureCorrelation


class FeatureExtractor(nn.Module):
    def __init__(self, encoder, tokenizer, sae, layer: int):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer = layer

    def forward(self, batch):
        if isinstance(batch, list) and isinstance(batch[0], str):
            batch = self.tokenize(batch)

        # Extract hidden states
        with torch.inference_mode():
            enc = self.encoder(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            )
            if isinstance(self.layer, float):
                layer = floor(len(enc["hidden_states"]) * self.layer)
            else:
                layer = self.layer

            hidden_state = enc["hidden_states"][layer]

        # Compute features activations
        B, S, D = hidden_state.shape
        feature_act = self.sae.forward(hidden_state.reshape(B * S, D)).reshape(B, S, -1)
        feature_act *= batch["attention_mask"].unsqueeze(-1)

        return feature_act

    def tokenize(self, batch):
        return self.tokenizer(
            batch,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.encoder.config.max_position_embeddings,
        )

    @classmethod
    def from_checkpoint(cls, sae_checkpoint: str):
        sae_checkpoint = Path(sae_checkpoint)
        config_path = sae_checkpoint.parent.parent.joinpath("config.json")
        sae_config = json.loads(config_path.read_text())

        # Load components
        encoder_args = sae_config["data"]["init_args"]
        encoder = load_encoder(encoder_args["name_or_path"])
        tokenizer = encoder_args["tokenizer"] or encoder_args["name_or_path"]
        tokenizer = load_tokenizer(tokenizer)

        # Load just the Sparse Autoencoder
        sae = SAE.load_from_checkpoint(
            sae_checkpoint, **sae_config["model"]["init_args"]
        ).sae
        layer = sae_config["data"]["init_args"]["layer"]

        return cls(encoder, tokenizer, sae, layer)


def collate_fn(batch, encoder_collate):
    encoder_input = {k: batch[k] for k in ["input_ids", "attention_mask"]}
    batch.update(encoder_collate(encoder_input))
    return batch


class FeaturePipeline:
    def __init__(
        self, sae_ckpt: str, dataset_path: str, features: Optional[list[str]] = None
    ):
        self.sae_ckpt = sae_ckpt
        self.dataset_path = dataset_path
        self.features = FeatureCollection.from_named(features or ["all"])

    @property
    def feature_names(self):
        return self.features.names

    def collate_fn(self, batch):
        tokens = [{k: x[k] for k in ["input_ids", "attention_mask"]} for x in batch]
        out = self.token_collator(tokens)
        for k in out.keys():
            if k not in ["input_ids", "attention_mask"]:
                out[k] = batch[k]
        return out

    def setup(self):
        self.miner = FeatureExtractor.from_checkpoint(self.sae_ckpt)

        # Load dataset
        dataset_path = Path(self.dataset_path)
        ds = load_dataset(
            "text",
            data_files={"val": str(dataset_path.joinpath("data/val/*.txt"))},
            keep_in_memory=False,
            streaming=True,
            save_infos=True,
        )
        ds = encode_molecules(ds, "text", encoding=MolEncoding.KEUKLE_SMILES)
        ds = ds.map(self.miner.tokenize, batched=True, input_columns="text")
        ds = ds.map(self.features, batched=False, input_columns="text")
        self.dataset = ds
        self.token_collator = DataCollatorWithPadding(self.miner.tokenizer, "longest")

    def iter(self, split: str = "val", **kwargs):
        dl = DataLoader(self.dataset[split], collate_fn=self.collate_fn, **kwargs)
        miner = self.miner.to("cuda")
        for batch in dl:
            x = {
                "input_ids": batch["input_ids"].to("cuda"),
                "attention_mask": batch["attention_mask"].to("cuda"),
            }
            f_act = miner(x)
            yield {
                "feature_activations": f_act.to("cpu"),
                "attention_mask": batch["attention_mask"],
                "proxy_activations": x["proxy_activations"],
            }
