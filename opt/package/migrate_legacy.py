import json
from pathlib import Path
from typing import Optional

from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormForMaskedLM,
)

from electrolyte_fm.models import (
    MISTExcessPhysics,
    MISTExcessPhysicsConfig,
    MISTFinetuned,
    MISTFinetunedConfig,
    MISTIonicConductivity,
    MISTIonicConductivityConfig,
    MISTMultiTask,
    MISTMultiTaskConfig,
)
from electrolyte_fm.models.excess_physics_model import pairwise_fusion
from electrolyte_fm.models.normalize import AbstractNormalizer
from electrolyte_fm.models.physics_task_heads import (
    ArrtheniusActivation,
    LinearExogenousEffect,
    VFTDecayTaskHead,
)
from electrolyte_fm.models.polynomials import LagrangePolynomial
from electrolyte_fm.models.prediction_task_head import PredictionTaskHead


def load_legacy_packaged_checkpoint(path: Path, model_class: Optional = None):
    """Load packaged checkpoint, handling both HF-compatible and legacy formats."""

    cfg = json.loads((path / "config.json").read_text())

    if model_class is None:
        model_class = cfg["architectures"][0]
    model_class = globals()[model_class]

    # Try HF from_pretrained first
    try:
        return model_class.from_pretrained(str(path), trust_remote_code=True)
    except Exception:
        pass

    # Fallback: legacy format - manually build and load weights
    enc_cfg = AutoConfig.for_model(cfg["encoder"]["model_type"]).from_dict(
        cfg["encoder"]
    )
    if hasattr(enc_cfg, "add_pooling_layer"):
        enc_cfg.add_pooling_layer = False
    encoder = AutoModel.from_config(enc_cfg)

    # Use the model class's from_components based on config structure
    if model_class == MISTIonicConductivity:
        tn_kwargs = cfg["task_network"].get(
            "kwargs", {"embed_dim": encoder.config.hidden_size}
        )
        model = model_class.from_components(
            encoder, VFTDecayTaskHead(**tn_kwargs), None, cfg.get("n_components", 38)
        )
    elif model_class == MISTExcessPhysics:
        state = load_file(str(path / "model.safetensors"))
        model = model_class(
            MISTExcessPhysicsConfig(
                encoder=cfg["encoder"],
                interactions=cfg.get("interactions", "difference"),
                num_control=cfg.get("num_control"),
                num_targets=cfg.get("num_targets", len(cfg.get("target_columns"))),
                temperature_dependence=cfg.get("temperature_dependence"),
                relative_excess=cfg.get("relative_excess", False),
                dropout=cfg.get("dropout", 0.1),
            )
        )

        model.encoder.encoder.load_state_dict(
            {
                k.replace("encoder.encoder.", ""): v
                for k, v in state.items()
                if k.startswith("encoder.encoder.")
            },
            strict=True,
        )
        model.pairwise_interaction.load_state_dict(
            {
                k.replace("pairwise_interaction.", ""): v
                for k, v in state.items()
                if k.startswith("pairwise_interaction.")
            },
            strict=True,
        )
        model.component_properties.load_state_dict(
            {
                k.replace("component_properties.", ""): v
                for k, v in state.items()
                if k.startswith("component_properties.")
            },
            strict=True,
        )
        model.excess_polynomial.load_state_dict(
            {
                k.replace("excess_polynomial.", ""): v
                for k, v in state.items()
                if k.startswith("excess_polynomial.")
            },
            strict=True,
        )
        model.transform.load_state_dict(
            {
                k.replace("transform.", ""): v
                for k, v in state.items()
                if k.startswith("transform.") and not k.startswith("excess_transform.")
            },
            strict=True,
        )
        model.excess_transform.load_state_dict(
            {
                k.replace("excess_transform.", ""): v
                for k, v in state.items()
                if k.startswith("excess_transform.")
            },
            strict=True,
        )

        return model
    elif model_class == MISTMultiTask:
        model = model_class.from_components(
            encoder,
            [
                PredictionTaskHead(t["embed_dim"], t["output_size"], t["dropout"])
                for t in cfg["task_networks"]
            ],
            [
                AbstractNormalizer.get(t["transform"], t["output_size"])
                for t in cfg["task_networks"]
            ],
            None,
            cfg.get("channels"),
        )
    else:  # MISTFinetuned
        tn = cfg["task_network"]
        model = model_class.from_components(
            encoder,
            PredictionTaskHead(tn["embed_dim"], tn["output_size"], tn["dropout"]),
            AbstractNormalizer.get(
                cfg["transform"]["class"], cfg["transform"]["num_outputs"]
            ),
            None,
            cfg.get("channels"),
        )

    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    return model


if __name__ == "__main___":
    model = load_legacy_packaged_checkpoint(
        "/nfs/turbo/coe-venkvis/abhutani/electrolyte-fm/solvent-properties",
        model_class=MISTMultiTask,
    )
    print(str(model))
