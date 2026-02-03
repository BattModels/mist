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
from .prediction_task_head import PredictionTaskHead
from .polynomials import LagrangePolynomial
from .mixture_model import TemperatureCondition
from .pairwise_fusion import pairwise_fusion
from .prod_finetune import (
    build_encoder_from_dict,
    annotate_prediction,
    maybe_get_annotated_channels,
)
from .model_utils import masked_mean_pool

from smirk import SmirkTokenizerFast

AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def resolve_tokenizer(self, tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    if getattr(self, "tokenizer", None) is not None:
        return self.tokenizer
    try:
        return AutoTokenizer.from_pretrained(
            self.name_or_path, use_fast=True, trust_remote_code=True
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            self.config._name_or_path, use_fast=True, trust_remote_code=True
        )


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

    def predict(
        self,
        batch: List[Dict[str, any]],
        return_dict: bool = True,
    ):
        tokenizer = resolve_tokenizer(self)
        collate = DataCollatorWithPadding(tokenizer)

        all_input_ids = [[] for _ in range(self.n_components)]
        all_attention_masks = [[] for _ in range(self.n_components)]
        all_compositions = [[] for _ in range(self.n_components)]
        all_temperatures = []

        for sample in batch:
            solvent_composition = sample["solvent_composition"]
            cation = sample["cation"]
            anion = sample["anion"]
            temperature = sample["temperature"]

            total_solvent_comp = sum(
                [v for comp_dict in solvent_composition for k, v in comp_dict.items()]
            )
            assert (
                total_solvent_comp < 1.0
            ), f"Total solvent mole fractions must be less than 1, got {total_solvent_comp}"

            salt_comp = 1.0 - total_solvent_comp

            components = list(solvent_composition)
            while len(components) < 3:
                components.append({"[H]": 0.0})  # Hydrogen as placeholder

            components.append({cation: salt_comp})
            components.append({anion: salt_comp})

            assert len(components) == 5, f"Expected 5 components, got {len(components)}"

            for i, component in enumerate(components):
                smiles = list(component.keys())[0]
                composition = list(component.values())[0]
                tok_output = tokenizer(smiles)
                all_input_ids[i].append(tok_output["input_ids"])
                all_attention_masks[i].append(tok_output["attention_mask"])
                all_compositions[i].append(composition)

            all_temperatures.append(temperature)

        output = {
            "temperature": torch.tensor(
                all_temperatures, dtype=torch.float32, device=self.device
            )
        }

        for i in range(self.n_components):
            batched = collate(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(all_input_ids[i], all_attention_masks[i])
                ]
            )
            output[f"input_ids_{i}"] = batched["input_ids"].to(self.device)
            output[f"attention_mask_{i}"] = batched["attention_mask"].to(self.device)
            output[f"composition_{i}"] = torch.tensor(
                all_compositions[i], dtype=torch.float32, device=self.device
            )

        with torch.inference_mode():
            pred, params = self(output, return_all=return_dict)

        if not return_dict:
            return pred.cpu()

        return {
            "ln conductivity [mS/cm]": pred.cpu(),
            "Ea": params["Ea"].cpu(),
            "Tg": params["Tg"].cpu(),
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

        # Configure Pairwise interaction model
        n_env = 0
        n_temperature_targets = 1
        if config.temperature_dependence == "arrhenius":
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

        self.pairwise_interaction = pairwise_fusion(
            config.interactions,
            config.encoder["hidden_size"],
            config.num_control * n_temperature_targets,
            n_targets=config.num_targets,
            n_env=n_env,
        )

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

        self.excess_polynomial = LagrangePolynomial(
            polynomial_order=config.num_control + 2,
            zero_endpoints=True,
        )

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

        # Load transform state dicts to preserve the registered buffers
        model.transform.load_state_dict(transform.state_dict())
        model.excess_transform.load_state_dict(excess_transform.state_dict())
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

        pw_coeffs = self.temperature_dependence(pw_coeffs, t_ij.view(B * I_, 1, 1))

        pw = x_t.view(B * I_, 1) * self.excess_polynomial(pw_coeffs, x_i)
        y_excess = pw.reshape(B, I_, -1).sum(dim=1)

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

        y_linear = self.transform.forward(y_linear)
        y_excess = self.excess_transform.forward(y_excess)
        y = y_linear + y_excess

        return y, y_linear, y_excess

    def predict(
        self,
        smiles_list: List[List[str]],
        composition: List[List[float]],
        temperature: List[float],
    ):
        tok = resolve_tokenizer(self, None)

        all_smiles = [smi for mixture in smiles_list for smi in mixture]

        inputs = tok(all_smiles, padding="longest", return_tensors="pt")

        batch_size = len(smiles_list)
        n_components = len(smiles_list[0])
        seq_len = inputs["input_ids"].shape[-1]

        input_ids = inputs["input_ids"].reshape(batch_size, n_components, seq_len)
        attention_mask = inputs["attention_mask"].reshape(
            batch_size, n_components, seq_len
        )

        composition_tensor = torch.tensor(composition, dtype=torch.float32)
        temperature_tensor = torch.tensor(temperature, dtype=torch.float32)

        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        composition_tensor = composition_tensor.to(device)
        temperature_tensor = temperature_tensor.to(device)

        with torch.no_grad():
            y, y_linear, y_excess = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                composition=composition_tensor,
                temperature=temperature_tensor,
            )

        return {
            "value": y.cpu(),
            "linear": y_linear.cpu(),
            "excess": y_excess.cpu(),
        }

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


class MISTMixturesConfig(PretrainedConfig):
    model_type = "mist_mixtures"

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        task_network: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None,
        n_components: int = 5,
        temperature_condition: str = "false",
        output_size: int = 2,
        channels: Optional[List[Dict[str, Any]]] = None,
        tokenizer_class: Optional[str] = "SmirkTokenizer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder or {}
        self.task_network = task_network or {
            "embed_dim": 768,
            "output_size": 2,
            "dropout": 0.1,
        }
        self.transform = transform or {"class": "Standardize", "num_outputs": 2}
        self.n_components = int(n_components)
        self.temperature_condition = temperature_condition
        self.output_size = int(output_size)
        self.channels = channels
        self.tokenizer_class = tokenizer_class


class MISTMixtures(PreTrainedModel):
    """
    Production-ready model for mixture property prediction with multiple targets.
    Supports various temperature conditioning strategies.
    """

    config_class = MISTMixturesConfig

    def __init__(self, config: MISTMixturesConfig):
        super().__init__(config)
        self.config = config
        self.encoder = build_encoder_from_dict(config.encoder)
        self.task_network = PredictionTaskHead(
            embed_dim=config.task_network.get(
                "embed_dim", self.encoder.config.hidden_size
            ),
            output_size=config.task_network.get("output_size", config.output_size),
            dropout=config.task_network.get("dropout", 0.1),
        )

        # Build transform
        self.transform = AbstractNormalizer.get(
            config.transform.get("class", "Standardize"),
            config.transform.get("num_outputs", config.output_size),
        )

        self.n_components = int(config.n_components)
        self.output_size = int(config.output_size)
        self.channels = config.channels

        if isinstance(config.temperature_condition, str):
            self.temperature_condition = TemperatureCondition(
                config.temperature_condition
            )
        else:
            self.temperature_condition = config.temperature_condition

        self.tokenizer = None
        self.post_init()

    @classmethod
    def from_components(
        cls,
        encoder: PreTrainedModel,
        task_network: nn.Module,
        transform: Any,
        tokenizer=None,
        n_components: int = 5,
        temperature_condition: str = "false",
        output_size: int = 2,
        channels: Optional[List[Dict[str, Any]]] = None,
    ) -> "MISTMixtures":
        """
        Create MISTMixtures from individual components.

        Args:
            encoder: Pretrained encoder model
            task_network: Task-specific prediction head
            transform: Normalization transform
            tokenizer: Tokenizer for encoding SMILES
            n_components: Maximum number of mixture components
            temperature_condition: Temperature conditioning strategy
            output_size: Number of output targets
            channels: List of channel metadata dictionaries
        """
        cfg = MISTMixturesConfig(
            encoder=encoder.config.to_dict(),
            task_network={
                "embed_dim": encoder.config.hidden_size,
                "output_size": output_size,
                "dropout": getattr(task_network, "dropout", 0.1),
            },
            transform=transform.to_config()
            if hasattr(transform, "to_config")
            else {"class": "Standardize", "num_outputs": output_size},
            n_components=n_components,
            temperature_condition=temperature_condition,
            output_size=output_size,
            channels=channels,
            tokenizer_class=(
                getattr(tokenizer, "__class__", type("T", (), {})).__name__
                if tokenizer
                else "SmirkTokenizer"
            ),
        )

        model = cls(cfg)
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        model.task_network.load_state_dict(task_network.state_dict())

        # Load transform state dict if available
        if hasattr(transform, "state_dict"):
            model.transform.load_state_dict(transform.state_dict())

        model.tokenizer = tokenizer
        return model

    def forward(
        self, batch: Dict[str, torch.Tensor], transform: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        mix_embedding = []
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)

            composition = batch[f"composition_{i}"].view(-1, 1)
            embedding_scaled = embedding * composition
            mix_embedding.append(embedding_scaled)

        # Sum weighted embeddings
        mix_embedding = torch.stack(mix_embedding, dim=1).sum(axis=1)

        if self.temperature_condition == TemperatureCondition.CONCAT:
            temperature_normalized = batch["temperature"].view(-1, 1) / 400.0
            mix_embedding = torch.hstack((temperature_normalized, mix_embedding))

        # Get predictions from task network
        pred_unscaled = self.task_network(mix_embedding)

        if transform:
            pred_scaled = self.transform.forward(pred_unscaled)
            return pred_scaled, mix_embedding

        return pred_unscaled, mix_embedding

    def forward(
        self,
        mixture: Dict[str, float] | List[Dict[str, float]],
        temperature: Optional[float | List[float]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        
      
        if isinstance(mixture, dict):
            mixtures = [mixture]
            single_prediction = True
        else:
            mixtures = mixture
            single_prediction = False

        # Convert temperature to list
        if temperature is not None:
            if isinstance(temperature, (int, float)):
                temperatures = [float(temperature)] * len(mixtures)
            else:
                temperatures = [float(t) for t in temperature]
        else:
            temperatures = None

        tokenizer = resolve_tokenizer(self)
        collator = DataCollatorWithPadding(tokenizer)

        batch_size = len(mixtures)

        smiles_list = []
        compositions = []
        for mix in mixtures:
            smiles = list(mix.keys())
            comps = list(mix.values())

            # Validate component count
            if len(smiles) > self.n_components:
                raise ValueError(
                    f"Too many components: {len(smiles)}, max {self.n_components}"
                )

            smiles_list.append(smiles)
            compositions.append(comps)

        batch = {}
        if self.temperature_condition != TemperatureCondition.NONE:
            if temperatures is None:
                raise ValueError("Temperature required for this model")
            assert len(temperatures) == batch_size, "Mismatch in temperature batch size"
            batch["temperature"] = torch.tensor(
                temperatures, dtype=torch.float32, device=self.device
            )

        for i in range(self.n_components):
            input_ids_list = []
            attention_mask_list = []
            comp_list = []

            for batch_idx in range(batch_size):
                if i < len(smiles_list[batch_idx]):
                    smi = smiles_list[batch_idx][i]
                    comp = compositions[batch_idx][i]
                else:
                    smi = "[H]"  # Dummy molecule for padding
                    comp = 0.0

                tok_output = tokenizer(smi)
                input_ids_list.append(tok_output["input_ids"])
                attention_mask_list.append(tok_output["attention_mask"])
                comp_list.append(comp)

            batched = collator(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(input_ids_list, attention_mask_list)
                ]
            )

            batch[f"input_ids_{i}"] = batched["input_ids"].to(self.device)
            batch[f"attention_mask_{i}"] = batched["attention_mask"].to(self.device)
            batch[f"composition_{i}"] = torch.tensor(
                comp_list, dtype=torch.float32, device=self.device
            )

        with torch.inference_mode():
            pred, embeddings = self(batch, transform=True)

        if single_prediction:
            pred = pred.squeeze(0)
            embeddings = embeddings.squeeze(0)

        pred_cpu = pred.cpu()

        if not return_dict:
            return pred_cpu

        if self.channels is not None:
            if single_prediction:
                pred_for_annotation = pred_cpu.unsqueeze(0)
            else:
                pred_for_annotation = pred_cpu

            annotated = annotate_prediction(
                pred_for_annotation, list(maybe_get_annotated_channels(self.channels))
            )

            if single_prediction:
                for key in annotated:
                    annotated[key]["value"] = annotated[key]["value"].squeeze(0)

            return annotated

        result = {
            "predictions": pred_cpu,
            "embeddings": embeddings.cpu(),
            "mixtures": mixtures,
        }

        if temperatures is not None:
            result["temperatures"] = (
                temperatures if not single_prediction else temperatures[0]
            )

        return result

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model configuration and weights."""
        cfg = MISTMixturesConfig(
            encoder=self.encoder.config.to_dict(),
            task_network={
                "embed_dim": self.encoder.config.hidden_size,
                "output_size": self.output_size,
                "dropout": getattr(self.task_network, "dropout", 0.1),
            },
            transform=self.transform.to_config()
            if hasattr(self.transform, "to_config")
            else {"class": "Standardize", "num_outputs": self.output_size},
            n_components=self.n_components,
            temperature_condition=self.temperature_condition.value
            if hasattr(self.temperature_condition, "value")
            else str(self.temperature_condition),
            output_size=self.output_size,
            channels=self.channels,
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
AutoConfig.register(MISTMixturesConfig.model_type, MISTMixturesConfig)
AutoModel.register(MISTMixturesConfig, MISTMixtures)
