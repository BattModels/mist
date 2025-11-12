# Finetuned Mixture Models for Inference

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from .normalize import AbstractNormalizer, Standardize
from .physics_task_heads import (
    VFTDecayTaskHead,
    ArrtheniusActivation,
    LinearExogenousEffect,
)
from .polynomials import LagrangePolynomial
from .excess_physics_model import pairwise_fusion
from .prod_finetune import build_encoder_from_dict
from .model_utils import masked_mean_pool

from smirk import SmirkTokenizerFast

AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def _resolve_tokenizer(self, tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    if getattr(self, "tokenizer", None) is not None:
        return self.tokenizer
    try:
        return AutoTokenizer.from_pretrained(self.name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(self.config._name_or_path, use_fast=True)


class MISTIonicConductivityConfig(PretrainedConfig):
    model_type = "mist_ionic_conductivity"

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        task_network: Optional[Dict[str, Any]] = None,
        n_components: int = 38,
        tokenizer_class: Optional[str] = "SmirkTokenizer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder or {}
        self.task_network = task_network or {"type": "VFTDecayTaskHead", "kwargs": {}}
        self.n_components = int(n_components)
        self.tokenizer_class = tokenizer_class


class MISTIonicConductivity(PreTrainedModel):
    config_class = MISTIonicConductivityConfig

    def __init__(self, config: MISTIonicConductivityConfig):
        super().__init__(config)
        self.encoder = build_encoder_from_dict(config.encoder)

        # Build task head
        tn_type = (config.task_network or {}).get("type", "VFTDecayTaskHead")
        tn_kwargs = (config.task_network or {}).get("kwargs", {})
        if tn_type == "VFTDecayTaskHead":
            self.task_network = VFTDecayTaskHead(**tn_kwargs)
        else:
            raise ValueError(f"Unknown task_network type: {tn_type}")

        self.n_components = int(config.n_components)
        self.tokenizer = None
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_network: nn.Module,
        tokenizer=None,
        n_components: int = 38,
    ) -> "MISTIonicConductivity":
        # try to minimal-config the head
        if isinstance(task_network, VFTDecayTaskHead):
            tn = {
                "type": "VFTDecayTaskHead",
                "kwargs": {"embed_dim": encoder.config.hidden_size},
            }
        else:
            raise ValueError(
                "Unsupported task head for MISTIonicConductivity.from_components"
            )

        cfg = MISTIonicConductivityConfig(
            encoder=encoder.config.to_dict(),
            task_network=tn,
            n_components=n_components,
            tokenizer_class=(
                getattr(tokenizer, "__class__", type("T", (), {})).__name__
                if tokenizer
                else "SmirkTokenizer"
            ),
        )
        model = cls(cfg)
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        model.task_network.load_state_dict(task_network.state_dict())
        model.tokenizer = tokenizer
        return model

    def forward(self, batch: Dict[str, torch.Tensor], return_all: bool = False):
        mix_embedding = None
        for i in range(self.n_components):
            enc = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state[:, 0, :]

            comp = batch[f"composition_{i}"].view(-1, 1)
            enc = enc * comp
            mix_embedding = enc if mix_embedding is None else (mix_embedding + enc)

        params = self.task_network(mix_embedding, batch["temperature"])
        pred_unscaled = params["conductivity"]
        alpha = params["alpha"]
        beta = params["beta"]
        lmbda = params["beta"]

        exponent = (-1.0 * alpha + batch["composition_4"]) / lmbda
        pred_decay = (1 - beta) * torch.exp(exponent) + beta
        pred = pred_unscaled * pred_decay

        pred = torch.where(batch["composition_4"] > alpha, pred, pred_unscaled)

        if return_all:
            return pred.view(-1, 1), params
        return pred.view(-1, 1), alpha

    def embed_mixture(
        self, smiles_list: List[List[str]], compositions: List[List[float]]
    ):
        tok = _resolve_tokenizer(self)
        batch_size = len(smiles_list)
        mixes = []
        with torch.inference_mode():
            for b in range(batch_size):
                mix_emb = None
                for i in range(min(len(smiles_list[b]), self.n_components)):
                    smi = smiles_list[b][i]
                    comp = torch.tensor(
                        compositions[b][i], dtype=torch.float32, device=self.device
                    )

                    t = tok([smi], return_tensors="pt", padding=True)
                    input_ids = t["input_ids"].to(self.device)
                    attention_mask = t["attention_mask"].to(self.device)
                    enc = self.encoder(
                        input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                    ).last_hidden_state[:, 0, :]
                    enc = enc * comp
                    mix_emb = enc if mix_emb is None else (mix_emb + enc)
                mixes.append(mix_emb)
        return torch.cat(mixes, dim=0).cpu()

    def predict(
        self,
        smiles_list: List[List[str]],
        compositions: List[List[float]],
        temperatures: List[float],
        salt_molarities: List[float],
        return_dict: bool = True,
        tokenizer=None,
    ):
        tok = _resolve_tokenizer(self, tokenizer)
        B = len(smiles_list)
        batch: Dict[str, torch.Tensor] = {
            "temperature": torch.tensor(
                temperatures, dtype=torch.float32, device=self.device
            ),
            "composition_4": torch.tensor(
                salt_molarities, dtype=torch.float32, device=self.device
            ),
        }

        # build per-component padded inputs
        for i in range(self.n_components):
            input_ids_list, attention_mask_list, comp_list = [], [], []
            for b in range(B):
                if i < len(smiles_list[b]):
                    smi = smiles_list[b][i]
                    comp = compositions[b][i]
                else:
                    smi = "[H]"
                    comp = 0.0
                tokens = tok([smi], return_tensors="pt", padding=True)
                input_ids_list.append(tokens["input_ids"].squeeze(0))
                attention_mask_list.append(tokens["attention_mask"].squeeze(0))
                comp_list.append(comp)

            collator = DataCollatorWithPadding(tok)
            collated = collator(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(input_ids_list, attention_mask_list)
                ]
            )
            batch[f"input_ids_{i}"] = collated["input_ids"].to(self.device)
            batch[f"attention_mask_{i}"] = collated["attention_mask"].to(self.device)
            batch[f"composition_{i}"] = torch.tensor(
                comp_list, dtype=torch.float32, device=self.device
            )

        with torch.inference_mode():
            pred, params = self(batch, return_all=return_dict)

        if not return_dict:
            return pred.cpu()
        return {
            "conductivity": pred.cpu(),
            "alpha": params["alpha"].cpu(),
            "beta": params["beta"].cpu(),
            "conductivity_unscaled": params["conductivity"].cpu(),
        }

    def save_pretrained(self, save_directory, **kwargs):
        tn = {
            "type": "VFTDecayTaskHead",
            "kwargs": {"embed_dim": self.encoder.config.hidden_size},
        }
        cfg = MISTIonicConductivityConfig(
            encoder=self.encoder.config.to_dict(),
            task_network=tn,
            n_components=self.n_components,
            tokenizer_class=(
                self.tokenizer.__class__.__name__
                if getattr(self, "tokenizer", None)
                else "SmirkTokenizer"
            ),
        )
        super().save_pretrained(save_directory, config=cfg, **kwargs)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(save_directory)


class MISTExcessPhysicsConfig(PretrainedConfig):
    model_type = "mist_excess_physics"

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        interactions: str = "difference",
        num_control: int = 3,
        num_targets: int = 1,
        temperature_dependence: Optional[str] = None,
        relative_excess: bool = False,
        dropout: float = 0.1,
        tokenizer_class: Optional[str] = "SmirkTokenizer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder or {}
        self.interactions = interactions
        self.num_control = num_control
        self.num_targets = num_targets
        self.temperature_dependence = temperature_dependence
        self.relative_excess = relative_excess
        self.dropout = dropout
        self.tokenizer_class = tokenizer_class


class MISTExcessPhysics(PreTrainedModel):
    config_class = MISTExcessPhysicsConfig

    def __init__(self, config: MISTExcessPhysicsConfig):
        super().__init__(config)
        self.config = config
        self.encoder = build_encoder_from_dict(config.encoder)

        # Temperature dependence setup
        n_temperature_targets = 1
        n_env = 0
        if config.temperature_dependence == "arrthenius":
            n_temperature_targets = 2
            self.temperature_dependence = ArrtheniusActivation()
        elif config.temperature_dependence == "concat":
            n_env = 1
            self.temperature_dependence = lambda x, t: x
        elif config.temperature_dependence == "locally-linear":
            n_temperature_targets = 2
            n_env = 1
            self.temperature_dependence = LinearExogenousEffect()
        else:
            self.temperature_dependence = lambda x, t: x

        # Pairwise interaction module
        self.pairwise_interaction = pairwise_fusion(
            config.interactions,
            self.encoder.config.hidden_size,
            config.num_control * n_temperature_targets,
            n_targets=config.num_targets,
            n_env=n_env,
        )

        # Component properties network
        self.component_properties = nn.Sequential(
            nn.Linear(
                self.encoder.config.hidden_size + n_env, self.encoder.config.hidden_size
            ),
            nn.Dropout(config.dropout),
            nn.SiLU(),
            nn.Linear(
                self.encoder.config.hidden_size,
                config.num_targets * n_temperature_targets,
            ),
        )

        # Excess polynomial
        self.excess_polynomial = LagrangePolynomial(
            polynomial_order=config.num_control + 2,
            zero_endpoints=True,
        )

        # Transforms
        self.transform = Standardize(num_outputs=config.num_targets)
        self.excess_transform = Standardize(num_outputs=config.num_targets)

        self.tokenizer = None
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        pairwise_interaction: nn.Module,
        component_properties: nn.Module,
        excess_polynomial: nn.Module,
        transform: Any,
        excess_transform: Any,
        tokenizer=None,
        interactions: str = "difference",
        num_control: int = 3,
        num_targets: int = 1,
        temperature_dependence: Optional[str] = None,
        relative_excess: bool = False,
        dropout: float = 0.1,
    ) -> "MISTExcessPhysics":
        cfg = MISTExcessPhysicsConfig(
            encoder=encoder.config.to_dict(),
            interactions=interactions,
            num_control=num_control,
            num_targets=num_targets,
            temperature_dependence=temperature_dependence,
            relative_excess=relative_excess,
            dropout=dropout,
            tokenizer_class=(
                getattr(tokenizer, "__class__", type("T", (), {})).__name__
                if tokenizer
                else "SmirkTokenizer"
            ),
        )
        model = cls(cfg)
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        model.pairwise_interaction.load_state_dict(pairwise_interaction.state_dict())
        model.component_properties.load_state_dict(component_properties.state_dict())
        model.excess_polynomial.load_state_dict(excess_polynomial.state_dict())
        model.transform = transform
        model.excess_transform = excess_transform
        model.tokenizer = tokenizer
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        composition: torch.Tensor,
        temperature: torch.Tensor,
    ):
        # Get component embedding vectors
        hs = self.encoder(
            input_ids.reshape(-1, input_ids.shape[-1]),
            attention_mask=attention_mask.reshape(-1, attention_mask.shape[-1]),
            return_dict=True,
            output_attentions=False,
        ).last_hidden_state.reshape(*input_ids.shape, -1)

        # Mean pool over tokens (B, C, L, E) -> (B, C, E)
        embs = masked_mean_pool(hs, attention_mask)
        y, y_linear, y_excess = self.compute_interactions(
            composition, embs, temperature
        )
        return y, y_linear, y_excess

    def compute_interactions(
        self, composition: torch.Tensor, embs: torch.Tensor, temperature: torch.Tensor
    ):
        # Compute Pairwise interactions
        B, C, E = embs.shape
        indices = torch.triu_indices(C, C, offset=1)
        I_ = indices.shape[1]
        e_i = embs[:, indices[0]].reshape(B * I_, E)
        e_j = embs[:, indices[1]].reshape(B * I_, E)
        t_ij = temperature.view(B, 1, 1).expand(-1, I_, -1).reshape(B * I_, 1)
        pw_coeffs = self.pairwise_interaction(e_i, e_j, t_ij)

        # Compute partial pairwise concentrations
        x_t = composition[:, indices[0]] + composition[:, indices[1]]
        x_t = x_t.clamp(min=0, max=1)
        x_i = composition[:, indices[0]] / x_t
        x_i = torch.where(x_i.abs() > 1e-8, x_i, torch.tensor(0.0).to(x_i))
        x_i = x_i.clamp(min=0, max=1)
        x_i = x_i.view(B * I_, 1)

        # Apply temperature dependence to coefficients
        pw_coeffs = self.temperature_dependence(pw_coeffs, t_ij.view(B * I_, 1, 1))

        # Evaluate interaction polynomials at compositions
        pw = x_t.view(B * I_, 1) * self.excess_polynomial(pw_coeffs, x_i)

        # Sum over interactions to get excess properties
        y_excess = pw.reshape(B, I_, -1).sum(dim=1)

        # Predict Pure Property (B, C, E) -> (B, C, T)
        if self.config.temperature_dependence in ["concat", "locally-linear"]:
            t_e = temperature.view(B, 1, 1).expand(-1, C, -1)
            y_target = self.component_properties(torch.cat([embs, t_e], dim=-1))
        else:
            y_target = self.component_properties(embs)

        y_target = self.temperature_dependence(y_target, temperature.view(B, 1, 1))

        # Linear Mixing (B, C, T) -> (B, T)
        y_linear = (y_target * composition.view(B, C, 1)).sum(dim=1)

        if self.config.relative_excess:
            y_excess *= y_linear

        # Transform to real-units
        y_linear = self.transform.forward(y_linear)
        y_excess = self.excess_transform.forward(y_excess)
        y = y_linear + y_excess

        return y, y_linear, y_excess

    def save_pretrained(self, save_directory, **kwargs):
        cfg = MISTExcessPhysicsConfig(
            encoder=self.encoder.config.to_dict(),
            interactions=self.config.interactions,
            num_control=self.config.num_control,
            num_targets=self.config.num_targets,
            temperature_dependence=self.config.temperature_dependence,
            relative_excess=self.config.relative_excess,
            dropout=self.config.dropout,
            tokenizer_class=(
                self.tokenizer.__class__.__name__
                if getattr(self, "tokenizer", None)
                else "SmirkTokenizer"
            ),
        )
        super().save_pretrained(save_directory, config=cfg, **kwargs)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(save_directory)


AutoConfig.register(MISTIonicConductivityConfig.model_type, MISTIonicConductivityConfig)
AutoModel.register(MISTIonicConductivityConfig, MISTIonicConductivity)
AutoConfig.register(MISTExcessPhysicsConfig.model_type, MISTExcessPhysicsConfig)
AutoModel.register(MISTExcessPhysicsConfig, MISTExcessPhysics)
