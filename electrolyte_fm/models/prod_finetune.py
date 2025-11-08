# Finetuned Models for Inference
import json
from pathlib import Path
import logging

import torch
from smirk import SmirkTokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding

from .prediction_task_head import PredictionTaskHead
from .normalize import AbstractNormalizer

AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def save_model(model, save_directory, safe_serialization=False):
    if safe_serialization:
        from safetensors.torch import save_model

        save_model(model, Path(save_directory, "model.safetensors"))
    else:
        torch.save(model.state_dict(), Path(save_directory, "model.pt"))


def load_model(model, save_directory):
    if (file := Path(save_directory, "model.safetensors")).is_file():
        from safetensors.torch import load_model as st_load_model

        unexpected, missing = st_load_model(model, file)
        if unexpected or missing:
            logging.warning(
                "Unexpected or missing tensors when loading %s, unexpected: %s missing: %s",
                file,
                unexpected,
                missing,
            )

    elif (file := Path(save_directory, "model.pt")).is_file():
        model.load_state_dict(torch.load(file, weights_only=True))
    else:
        raise RuntimeError("No model found")


def maybe_get_annotated_channels(channels: list):
    for chn in channels:
        if isinstance(chn, str):
            yield {"name": chn, "description": None, "unit": None}
        else:
            yield chn


def annotate_prediction(y: torch.Tensor, channels: list[dict[str, str]]) -> dict:
    out = {}
    for idx, chn in enumerate(channels):
        channel_info = {f: v for f, v in chn.items() if f != "name"}
        out[chn["name"]] = {"value": y[:, idx], **channel_info}
    return out


class MISTFinetuned(torch.nn.Module):
    def __init__(self, encoder, task_network, transform, tokenizer, channels=None):
        super().__init__()
        self.encoder = encoder
        self.task_network = task_network
        self.transform = transform
        self.tokenizer = tokenizer
        self.channels = channels

    def forward(self, input_ids, attention_mask=None):
        hs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        y = self.task_network(hs)
        return self.transform.forward(y)

    def save_pretrained(self, save_directory, safe_serialization=False):
        config = {
            "architectures": [
                self.__class__.__name__,
            ],
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "encoder": self.encoder.config.to_diff_dict(),
            "task_network": {
                "embed_dim": self.encoder.config.hidden_size,
                "output_size": self.task_network.final.out_features,
                "dropout": self.task_network.dropout1.p,
            },
            "transform": self.transform.to_config(),
            "channels": self.channels,
        }

        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))
        save_model(self, save_directory, safe_serialization)

    def embed(self, smi: list[str]):
        batch = self.tokenizer(smi)
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        batch = collate_fn(batch)
        input_ids = batch["input_ids"].to(self.encoder.device)
        attention_mask = batch["attention_mask"].to(self.encoder.device)
        with torch.inference_mode():
            hs = self.encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]

        return hs.to("cpu")

    def predict(self, smi: list[str], return_dict=True):
        batch = self.tokenizer(smi)
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        batch = collate_fn(batch)
        batch = {
            "input_ids": batch["input_ids"].to(self.encoder.device),
            "attention_mask": batch["attention_mask"].to(self.encoder.device),
        }
        with torch.inference_mode():
            out = self(**batch).cpu()

        if self.channels is None or not return_dict:
            return out

        return annotate_prediction(out, self.channels)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "MISTFinetuned":
        config = json.loads(Path(name_or_path, "config.json").read_text())
        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        task_network = PredictionTaskHead(**config["task_network"])
        transform = AbstractNormalizer.get(
            config["transform"]["class"], config["transform"]["num_outputs"]
        )

        # Instantiate model
        model = cls(encoder, task_network, transform, config["channels"])
        load_model(model, name_or_path)
        return model


class MISTMultiTask(torch.nn.Module):
    def __init__(self, encoder, task_networks, transforms, tokenizer, channels=None):
        super().__init__()
        self.encoder = encoder
        self.task_networks = torch.nn.ModuleList(task_networks)
        self.transforms = torch.nn.ModuleList(transforms)
        self.tokenizer = tokenizer
        assert len(self.task_networks) == len(self.transforms)
        self.channels = channels

    def forward(self, input_ids, attention_mask=None):
        hs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        out = []
        for tn, tf in zip(self.task_networks, self.transforms):
            out.append(tf.forward(tn(hs)))

        return torch.cat(out, dim=-1)

    def predict(self, smi: list[str]):
        batch = self.tokenizer(smi)
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        batch = collate_fn(batch)
        batch = {
            "input_ids": batch["input_ids"].to(self.encoder.device),
            "attention_mask": batch["attention_mask"].to(self.encoder.device),
        }
        with torch.inference_mode():
            out = self(**batch).cpu()

        if self.channels is None:
            return out
        return annotate_prediction(out, self.channels)

    def embed(self, smi: list[str]):
        batch = self.tokenizer(smi)
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        batch = collate_fn(batch)
        input_ids = batch["input_ids"].to(self.encoder.device)
        attention_mask = batch["attention_mask"].to(self.encoder.device)
        with torch.inference_mode():
            hs = self.encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]

        return hs.to("cpu")

    def save_pretrained(self, save_directory, safe_serialization=False):
        config = {
            "architectures": [
                self.__class__.__name__,
            ],
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "encoder": self.encoder.config.to_diff_dict(),
            "task_networks": [
                {
                    "embed_dim": self.encoder.config.hidden_size,
                    "output_size": tn.final.out_features,
                    "transform": tf.__class__.__name__,
                    "dropout": tn.dropout1.p,
                }
                for tn, tf in zip(self.task_networks, self.transforms)
            ],
            "channels": self.channels,
        }
        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))
        save_model(self, save_directory, safe_serialization)

    @classmethod
    def from_pretrained(self, save_directory: str):
        config = json.loads(Path(save_directory, "config.json").read_text())
        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)

        task_networks = []
        transforms = []
        for tc in config["task_networks"]:
            transforms.append(
                AbstractNormalizer.get(tc.pop("transform"), tc["output_size"])
            )
            task_networks.append(PredictionTaskHead(**tc))

        channels = list(maybe_get_annotated_channels(config["channels"]))
        model = MISTMultiTask(encoder, task_networks, transforms, tokenizer, channels)
        load_model(model, save_directory)
        return model
