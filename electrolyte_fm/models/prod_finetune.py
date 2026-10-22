# Finetuned Models for Inference

import logging
from pathlib import Path
from typing import Any

import torch
from smirk import SmirkTokenizerFast
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
)

from .normalize import AbstractNormalizer
from .position_ids import create_position_ids
from .prediction_task_head import PredictionTaskHead

AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def maybe_get_annotated_channels(channels: list[Any]):
    for chn in channels:
        if isinstance(chn, str):
            yield {"name": chn, "description": None, "unit": None}
        else:
            yield chn


def annotate_prediction(
    y: torch.Tensor, channels: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
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

    kwargs = (
        {"add_pooling_layer": enc_dict.get("add_pooling_layer", False)}
        if enc_cfg.model_type in {"bert", "roberta", "roberta-prelayernorm"}
        else {}
    )
    token_padding_idx = enc_cfg.pad_token_id
    if hasattr(enc_cfg, "position_padding_idx"):
        # RoBERTa constructs both embeddings with pad_token_id. Tokenizer IDs
        # can exceed the position table, so construct with the saved positional
        # offset, then restore the independent token padding ID.
        enc_cfg.pad_token_id = enc_cfg.position_padding_idx
    encoder = AutoModel.from_config(enc_cfg, **kwargs)
    if hasattr(enc_cfg, "position_padding_idx"):
        enc_cfg.pad_token_id = token_padding_idx
        encoder.get_input_embeddings().padding_idx = token_padding_idx
        embeddings = getattr(encoder, "embeddings", None)
        positions = getattr(embeddings, "position_embeddings", None)
        if positions is not None:
            # Token padding and the pretrained positional offset are separate.
            embeddings.padding_idx = enc_cfg.position_padding_idx
            positions.padding_idx = enc_cfg.position_padding_idx
    return encoder


def encoder_config(encoder):
    config = encoder.config.to_dict()
    if hasattr(encoder, "pooler"):
        config["add_pooling_layer"] = encoder.pooler is not None
    return config


def encode_inputs(encoder, input_ids, attention_mask=None, position_ids=None):
    """Preserve legacy positions unless the checkpoint records a separate offset."""
    kwargs = {}
    if position_ids is None and hasattr(encoder.config, "position_padding_idx"):
        position_ids = create_position_ids(
            input_ids, encoder.config.pad_token_id, encoder.config.position_padding_idx
        )
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return encoder(input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state


def batched_inference(model, smi, tokenizer, batch_size, *, embeddings=False):
    """Tokenize without changing SMILES, sharing batching across production models."""
    tokenizer = model._resolve_tokenizer(tokenizer)
    if tokenizer is None:
        raise ValueError("Prediction requires a tokenizer")
    smi = list(smi)
    if batch_size is None:
        batch_size = 8 if model.encoder.config.hidden_size > 1024 else 64
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    outputs = []
    start = 0
    collate = DataCollatorWithPadding(tokenizer)
    while start < len(smi):
        try:
            batch = collate(tokenizer(smi[start : start + batch_size]))
            ids = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)
            with torch.inference_mode():
                out = (
                    encode_inputs(model.encoder, ids, mask)[:, 0, :]
                    if embeddings
                    else model(ids, attention_mask=mask)
                ).cpu()
            # NumPy and PythonCall cannot consume torch.bfloat16 tensors.
            outputs.append(out.float() if out.dtype == torch.bfloat16 else out)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
            continue
        start += batch_size
    if outputs:
        return torch.cat(outputs)
    width = (
        model.encoder.config.hidden_size
        if embeddings
        else (
            sum(head["output_size"] for head in model.config.task_networks)
            if hasattr(model.config, "task_networks")
            else model.config.task_network["output_size"]
        )
    )
    dtype = torch.float32 if model.dtype == torch.bfloat16 else model.dtype
    return torch.empty((0, width), dtype=dtype)


def load_finetuned(checkpoint, *, tokenizer=None, local_files_only=False):
    """Load a MIST Hugging Face release or a trusted local Lightning checkpoint."""
    if Path(checkpoint).suffix == ".ckpt":
        return MISTFinetuned.from_lightning_checkpoint(
            checkpoint, tokenizer=tokenizer, local_files_only=local_files_only
        )
    config, _ = PretrainedConfig.get_config_dict(
        checkpoint, local_files_only=local_files_only
    )
    model_class = {
        "mist_finetuned": MISTFinetuned,
        "mist_multitask": MISTMultiTask,
    }.get(config.get("model_type"))
    if model_class is None:
        raise ValueError(f"Not a packaged MIST task model: {config.get('model_type')}")
    if tokenizer is None or isinstance(tokenizer, (str, Path)):
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint if tokenizer is None else tokenizer,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    model = model_class.from_pretrained(
        checkpoint, tokenizer=tokenizer, local_files_only=local_files_only
    ).eval()
    model.encoding = getattr(model.config, "encoding", "smiles-kekule")
    return model


class MISTFinetunedConfig(PretrainedConfig):
    """HF config for a single-task MIST wrapper."""

    model_type = "mist_finetuned"

    def __init__(
        self,
        encoder: dict[str, Any] | None = None,
        task_network: dict[str, Any] | None = None,
        transform: dict[str, Any] | None = None,
        channels: list[dict[str, Any]] | None = None,
        tokenizer_class: str | None = "SmirkTokenizer",
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

    def __init__(self, config: MISTFinetunedConfig, *, tokenizer=None, components=None):
        super().__init__(config)
        if components is not None:
            self.encoder, self.task_network, self.transform = components
            self.channels = config.channels
            self.tokenizer = tokenizer
            return
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
        self.tokenizer = self._resolve_tokenizer(tokenizer)
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_network: nn.Module,
        transform: Any,
        tokenizer: Any | None = None,
        channels: list[dict[str, Any]] | None = None,
        *,
        copy: bool = True,
    ) -> "MISTFinetuned":
        cfg = MISTFinetunedConfig(
            encoder=encoder_config(encoder),
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
        if not copy:
            # Conversion can transfer existing components without allocating a
            # second encoder (important for the 1.8B model).
            return cls(
                cfg, tokenizer=tokenizer, components=(encoder, task_network, transform)
            )
        model = cls(cfg, tokenizer=tokenizer)
        # load component weights
        model.encoder.load_state_dict(encoder.state_dict(), strict=True)
        model.task_network.load_state_dict(task_network.state_dict())
        model.transform.load_state_dict(transform.state_dict())
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_lightning_checkpoint(
        cls, checkpoint, *, tokenizer=None, local_files_only=False
    ):
        """Convert a trusted training checkpoint without copying its encoder.

        Lightning files contain pickled training metadata; only load checkpoints
        you trust. HF inference does not require Lightning to be installed.
        """
        from electrolyte_fm.models.lm_finetuning import LMFinetuning
        from electrolyte_fm.utils.tokenizer import load_tokenizer

        trained = LMFinetuning.load_from_checkpoint(
            checkpoint, map_location="cpu", weights_only=False
        )
        if tokenizer is None:
            tokenizer = trained.hparams.get("tokenizer") or trained.hparams.encoder_ckpt
        if isinstance(tokenizer, (str, Path)):
            tokenizer = load_tokenizer(
                str(tokenizer), local_files_only=local_files_only
            )
        return cls.from_components(
            trained.encoder,
            trained.task_network,
            trained.transform,
            tokenizer=tokenizer,
            channels=trained.hparams.target_columns,
            copy=False,
        ).eval()

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hs = encode_inputs(self.encoder, input_ids, attention_mask, position_ids)
        y = self.task_network(hs)
        return self.transform.forward(y)

    def _resolve_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if getattr(self, "tokenizer", None) is not None:
            return self.tokenizer

        if self.name_or_path and "/" in self.name_or_path:
            try:
                return AutoTokenizer.from_pretrained(
                    self.name_or_path, use_fast=True, trust_remote_code=True
                )
            except (OSError, ValueError, TypeError) as error:
                logging.getLogger(__name__).debug(
                    "Tokenizer discovery failed: %s", error
                )

        if (
            hasattr(self.config, "_name_or_path")
            and self.config._name_or_path
            and "/" in self.config._name_or_path
        ):
            try:
                return AutoTokenizer.from_pretrained(
                    self.config._name_or_path, use_fast=True, trust_remote_code=True
                )
            except (OSError, ValueError, TypeError) as error:
                logging.getLogger(__name__).debug(
                    "Tokenizer discovery failed: %s", error
                )

        return None

    def embed(self, smi: list[str], tokenizer=None, *, batch_size=None):
        return batched_inference(self, smi, tokenizer, batch_size, embeddings=True)

    def predict(
        self,
        smi: list[str],
        return_dict: bool = True,
        tokenizer=None,
        *,
        batch_size=None,
    ):
        out = batched_inference(self, smi, tokenizer, batch_size)
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
        encoder: dict[str, Any] | None = None,
        task_networks: list[dict[str, Any]] | None = None,
        transforms: list[dict[str, Any]] | None = None,
        channels: list[dict[str, Any]] | None = None,
        tokenizer_class: str | None = "SmirkTokenizer",
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

    def __init__(self, config: MISTMultiTaskConfig, *, tokenizer=None, components=None):
        super().__init__(config)
        if components is not None:
            self.encoder, task_networks, transforms = components
            self.task_networks = nn.ModuleList(task_networks)
            self.transforms = nn.ModuleList(transforms)
            self.channels = config.channels
            self.tokenizer = tokenizer
            return
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

        assert len(self.task_networks) == len(self.transforms), (
            "task_networks and transforms must align"
        )
        self.channels = config.channels
        self.tokenizer = self._resolve_tokenizer(tokenizer)
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_networks: list[nn.Module],
        transforms: list[Any],
        tokenizer: Any | None = None,
        channels: list[dict[str, Any]] | None = None,
        *,
        copy: bool = True,
    ) -> "MISTMultiTask":
        if len(task_networks) != len(transforms):
            raise ValueError("task_networks and transforms must align")
        cfg = MISTMultiTaskConfig(
            encoder=encoder_config(encoder),
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
        if not copy:
            return cls(
                cfg,
                tokenizer=tokenizer,
                components=(encoder, task_networks, transforms),
            )
        model = cls(cfg, tokenizer=tokenizer)
        model.encoder.load_state_dict(encoder.state_dict(), strict=True)
        for dst, src in zip(model.task_networks, task_networks):
            dst.load_state_dict(src.state_dict())
        for dst, src in zip(model.transforms, transforms):
            dst.load_state_dict(src.state_dict())
        model.tokenizer = tokenizer
        return model

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hs = encode_inputs(self.encoder, input_ids, attention_mask, position_ids)
        outs = []
        for tn, tf in zip(self.task_networks, self.transforms):
            outs.append(tf.forward(tn(hs)))
        return torch.cat(outs, dim=-1)

    def _resolve_tokenizer(self, tokenizer=None):
        return MISTFinetuned._resolve_tokenizer(self, tokenizer)

    def predict(
        self, smi: list[str], tokenizer=None, *, return_dict=True, batch_size=None
    ):
        out = batched_inference(self, smi, tokenizer, batch_size)
        if self.channels is None or not return_dict:
            return out
        return annotate_prediction(out, maybe_get_annotated_channels(self.channels))

    def embed(self, smi: list[str], tokenizer=None, *, batch_size=None):
        return batched_inference(self, smi, tokenizer, batch_size, embeddings=True)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(save_directory)


AutoConfig.register(MISTFinetunedConfig.model_type, MISTFinetunedConfig)
AutoModel.register(MISTFinetunedConfig, MISTFinetuned)

AutoConfig.register(MISTMultiTaskConfig.model_type, MISTMultiTaskConfig)
AutoModel.register(MISTMultiTaskConfig, MISTMultiTask)
