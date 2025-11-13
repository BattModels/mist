# Finetuned Models for Inference

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PretrainedConfig,
)

from smirk import SmirkTokenizerFast
from .prediction_task_head import PredictionTaskHead
from .normalize import AbstractNormalizer

AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def maybe_get_annotated_channels(channels: List[Any]):
    for chn in channels:
        if isinstance(chn, str):
            yield {"name": chn, "description": None, "unit": None}
        else:
            yield chn


def annotate_prediction(
    y: torch.Tensor, channels: List[Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for idx, chn in enumerate(channels):
        channel_info = {f: v for f, v in chn.items() if f != "name"}
        out[chn["name"]] = {"value": y[:, idx], **channel_info}
    return out


def build_encoder_from_dict(enc_dict):
    if "model_type" in enc_dict:
        cfg_cls = AutoConfig.for_model(enc_dict["model_type"])
        enc_cfg = cfg_cls.from_dict(enc_dict, strict=False)
    elif "_name_or_path" in enc_dict:
        enc_cfg = AutoConfig.from_pretrained(enc_dict["_name_or_path"], strict=False)
    else:
        raise KeyError("Encoder config is missing 'model_type' and '_name_or_path.")

    # Ensure pooling layer is disabled to match saved checkpoints
    if hasattr(enc_cfg, "add_pooling_layer"):
        enc_cfg.add_pooling_layer = False

    return AutoModel.from_config(enc_cfg)


class MISTFinetunedConfig(PretrainedConfig):
    """HF config for a single-task MIST wrapper."""

    model_type = "mist_finetuned"

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        task_network: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None,
        channels: Optional[List[Dict[str, Any]]] = None,
        tokenizer_class: Optional[str] = "SmirkTokenizer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder or {}
        self.task_network = task_network or {}
        self.transform = transform or {}
        self.channels = channels
        self.tokenizer_class = tokenizer_class


class MISTFinetuned(PreTrainedModel):
    config_class = MISTFinetunedConfig

    def __init__(self, config: MISTFinetunedConfig):
        super().__init__(config)
        self.encoder = build_encoder_from_dict(config.encoder)

        tn = config.task_network
        self.task_network = PredictionTaskHead(
            embed_dim=tn["embed_dim"],
            output_size=tn["output_size"],
            dropout=tn["dropout"],
        )
        self.transform = AbstractNormalizer.get(
            config.transform["class"], config.transform["num_outputs"]
        )
        self.channels = config.channels
        self.tokenizer = None
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_network: nn.Module,
        transform: Any,
        tokenizer: Optional[Any] = None,
        channels: Optional[List[Dict[str, Any]]] = None,
    ) -> "MISTFinetuned":
        cfg = MISTFinetunedConfig(
            encoder=encoder.config.to_dict(),
            task_network={
                "embed_dim": encoder.config.hidden_size,
                "output_size": task_network.final.out_features,
                "dropout": task_network.dropout1.p,
            },
            transform=transform.to_config(),
            channels=channels,
            tokenizer_class=(
                getattr(tokenizer, "__class__", type("T", (), {})).__name__
                if tokenizer
                else "SmirkTokenizer"
            ),
        )
        model = cls(cfg)
        # load component weights
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        model.task_network.load_state_dict(task_network.state_dict())
        model.transform.load_state_dict(transform.state_dict())
        model.tokenizer = tokenizer
        return model

    def forward(self, input_ids, attention_mask=None):
        hs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        y = self.task_network(hs)
        return self.transform.forward(y)

    def _resolve_tokenizer(self, tokenizer):
        if tokenizer is not None:
            return tokenizer
        if getattr(self, "tokenizer", None) is not None:
            return self.tokenizer
        try:
            return AutoTokenizer.from_pretrained(self.name_or_path, use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.config._name_or_path, use_fast=True
            )

    def embed(self, smi: List[str], tokenizer=None):
        tok = self._resolve_tokenizer(tokenizer)
        batch = tok(smi)
        batch = DataCollatorWithPadding(tok)(batch)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.inference_mode():
            hs = self.encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        return hs.to("cpu")

    def predict(self, smi: List[str], return_dict: bool = True, tokenizer=None):
        tok = self._resolve_tokenizer(tokenizer)
        batch = tok(smi)
        batch = DataCollatorWithPadding(tok)(batch)
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self(**inputs).cpu()
        if self.channels is None or not return_dict:
            return out
        return annotate_prediction(out, maybe_get_annotated_channels(self.channels))

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(save_directory)


class MISTMultiTaskConfig(PretrainedConfig):
    """HuggingFace config for a multi-task MIST wrapper."""

    model_type = "mist_multitask"

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        task_networks: Optional[List[Dict[str, Any]]] = None,
        transforms: Optional[List[Dict[str, Any]]] = None,
        channels: Optional[List[Dict[str, Any]]] = None,
        tokenizer_class: Optional[str] = "SmirkTokenizer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder or {}
        self.task_networks = task_networks or []
        self.transforms = transforms or []
        self.channels = channels
        self.tokenizer_class = tokenizer_class


class MISTMultiTask(PreTrainedModel):
    config_class = MISTMultiTaskConfig

    def __init__(self, config: MISTMultiTaskConfig):
        super().__init__(config)
        self.encoder = build_encoder_from_dict(config.encoder)

        self.task_networks = nn.ModuleList(
            [
                PredictionTaskHead(
                    embed_dim=tn["embed_dim"],
                    output_size=tn["output_size"],
                    dropout=tn["dropout"],
                )
                for tn in config.task_networks
            ]
        )
        self.transforms = nn.ModuleList(
            [
                AbstractNormalizer.get(tf_cfg["class"], tf_cfg["num_outputs"])
                for tf_cfg in config.transforms
            ]
        )

        assert len(self.task_networks) == len(
            self.transforms
        ), "task_networks and transforms must align"
        self.channels = config.channels
        self.tokenizer = None
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_networks: List[nn.Module],
        transforms: List[Any],
        tokenizer: Optional[Any] = None,
        channels: Optional[List[Dict[str, Any]]] = None,
    ) -> "MISTMultiTask":
        cfg = MISTMultiTaskConfig(
            encoder=encoder.config.to_dict(),
            task_networks=[
                {
                    "embed_dim": encoder.config.hidden_size,
                    "output_size": tn.final.out_features,
                    "dropout": tn.dropout1.p,
                }
                for tn in task_networks
            ],
            transforms=[tf.to_config() for tf in transforms],
            channels=channels,
            tokenizer_class=(
                getattr(tokenizer, "__class__", type("T", (), {})).__name__
                if tokenizer
                else "SmirkTokenizer"
            ),
        )
        model = cls(cfg)
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        for dst, src in zip(model.task_networks, task_networks):
            dst.load_state_dict(src.state_dict())
        for dst, src in zip(model.transforms, transforms):
            dst.load_state_dict(src.state_dict())
        model.tokenizer = tokenizer
        return model

    def forward(self, input_ids, attention_mask=None):
        hs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        outs = []
        for tn, tf in zip(self.task_networks, self.transforms):
            outs.append(tf.forward(tn(hs)))
        return torch.cat(outs, dim=-1)

    def _resolve_tokenizer(self, tokenizer):
        if tokenizer is not None:
            return tokenizer
        if getattr(self, "tokenizer", None) is not None:
            return self.tokenizer
        try:
            return AutoTokenizer.from_pretrained(self.name_or_path, use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.config._name_or_path, use_fast=True
            )

    def predict(self, smi: List[str], tokenizer=None):
        tok = self._resolve_tokenizer(tokenizer)
        batch = tok(smi)
        batch = DataCollatorWithPadding(tok)(batch)
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self(**inputs).cpu()
        if self.channels is None:
            return out
        return annotate_prediction(out, self.channels)

    def embed(self, smi: List[str], tokenizer=None):
        tok = self._resolve_tokenizer(tokenizer)
        batch = tok(smi)
        batch = DataCollatorWithPadding(tok)(batch)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.inference_mode():
            hs = self.encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        return hs.to("cpu")

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(save_directory)


AutoConfig.register(MISTFinetunedConfig.model_type, MISTFinetunedConfig)
AutoModel.register(MISTFinetunedConfig, MISTFinetuned)

AutoConfig.register(MISTMultiTaskConfig.model_type, MISTMultiTaskConfig)
AutoModel.register(MISTMultiTaskConfig, MISTMultiTask)
