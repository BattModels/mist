#!/usr/bin/env python
# Script to package MIST models for downstream usage
#
# Usage:
#   python -m opt.package --help

from __future__ import annotations
import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Iterable, Type, Optional, List, Tuple, Set
import typer
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts, get_ckpt_tokenizer
from electrolyte_fm.models.lm_finetuning import load_encoder
from electrolyte_fm.models import (
    MISTFinetunedConfig,
    MISTFinetuned,
    MISTIonicConductivityConfig,
    MISTIonicConductivity,
    MISTMultiTaskConfig,
    MISTMultiTask,
    MISTExcessPhysicsConfig,
    MISTExcessPhysics,
    MISTMixturesConfig,
    MISTMixtures,
)
from electrolyte_fm.models.mixture_model import TemperatureCondition
from electrolyte_fm.utils.tokenizer import load_tokenizer
from electrolyte_fm.models.prediction_task_head import PredictionTaskHead
from electrolyte_fm.models.normalize import (
    AbstractNormalizer,
    Standardize,
    PowerTransform,
    LogTransform,
    MaxScaleTransform,
    IdentityTransform,
)
from electrolyte_fm.models.physics_task_heads import (
    VFTDecayTaskHead,
    ArrtheniusActivation,
    LinearExogenousEffect,
)
from electrolyte_fm.models.polynomials import LagrangePolynomial
from electrolyte_fm.models.pairwise_fusion import pairwise_fusion
from .utils import (
    name_model,
    get_best_ckpt,
    create_save_directory,
    ckpt_id,
    save_tokenizer,
    create_tar_gz,
)
from .migrate_legacy import load_legacy_packaged_checkpoint
from .write_model_class import write_modeling_module

cli = typer.Typer()
logging.basicConfig(level=logging.INFO)
L = logging.getLogger("package")
template = Path(__file__).parent / "modeling_mist.py.j2"


def patch_auto_map(
    save_dir: Path,
    *,
    module_basename: str,
    model_type: str,
    architecture_name: str,
    config_class_name: str,
    model_class_name: str,
):
    cfg_path = save_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    cfg["model_type"] = model_type
    cfg["architectures"] = [architecture_name]
    cfg["auto_map"] = {
        "AutoConfig": f"{module_basename}.{config_class_name}",
        "AutoModel": f"{module_basename}.{model_class_name}",
    }

    cfg_path.write_text(json.dumps(cfg, indent=2))
    L.info("Patched auto_map in %s", cfg_path)


def maybe_best_ckpt(path: Path) -> Path:
    return get_best_ckpt(path) if path.joinpath("config.json").is_file() else path


def read_training_config(ckpt: Path) -> dict:
    cfg_path = ckpt / "config.json"
    return json.loads(cfg_path.read_text())


def save_with_tokenizer(model, save_dir: Path, safe: bool = True, tokenizer=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.config, "add_pooling_layer"):
        model.config.add_pooling_layer = False
    model.save_pretrained(save_dir, safe_serialization=safe)
    tok = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    if tok is not None:
        tok.save_pretrained(save_dir)
    else:
        try:
            save_tokenizer(save_dir, tok)
        except Exception:
            pass


def load_model(ckpt: Path, model_class: Optional = None):
    try:
        ckpt = maybe_best_ckpt(ckpt)
        model = SaveConfigWithCkpts.load(ckpt, strict=False)
    except FileNotFoundError:
        L.warning("Attempting load from safetensors export")
        model = load_legacy_packaged_checkpoint(ckpt, model_class)
    # update ckpt path if maybe_best_ckpt
    return model, ckpt


@cli.command()
def pretrained(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """
    Export a pretrained encoder.
    """
    model, best_ckpt = load_model(ckpt)

    tag = name_model(
        model,
        template=name or "mist-{model_size}-{ckpt}",
        ckpt=ckpt_id(ckpt),
    )
    save_dir = create_save_directory(tag, best_ckpt)
    train_cfg = read_training_config(ckpt)
    tokenizer = train_cfg.get("data")
    if tokenizer:
        tokenizer = load_tokenizer(tokenizer["init_args"]["tokenizer"])
    else:
        tokenizer = load_tokenizer("smirk")
    if hasattr(model, "model"):
        save_with_tokenizer(model.model, save_dir, safe=safe, tokenizer=tokenizer)
    else:
        save_with_tokenizer(model, save_dir, safe=safe, tokenizer=tokenizer)
    L.info("Saved pretrained model to %s", save_dir)
    create_tar_gz(save_dir)


def export_finetuned(ckpt: Path) -> MISTFinetuned:
    bundle, best_ckpt = load_model(ckpt, model_class="MISTFinetuned")
    train_cfg = read_training_config(ckpt)
    tokenizer = train_cfg.get("data")
    if tokenizer:
        tokenizer = load_tokenizer(tokenizer["init_args"]["tokenizer"])
    else:
        tokenizer = load_tokenizer("smirk")

    # Try to get channels from training config or packaged model config
    try:
        channels = train_cfg["data"]["init_args"].get("target_columns")
    except KeyError:
        # If loading from already-packaged model, channels are at top level
        channels = train_cfg.get("channels")

    model = MISTFinetuned.from_components(
        encoder=bundle.encoder,
        task_network=bundle.task_network,
        transform=bundle.transform,
        tokenizer=tokenizer,
        channels=channels,
    )
    return model, best_ckpt


@cli.command()
def finetuned(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """
    Export a finetuned model with embedded remote code.
    """
    model, best_ckpt = export_finetuned(ckpt)

    tag = name_model(
        model,
        template=name or "mist-{model_size}-{ckpt}",
        ckpt=ckpt_id(best_ckpt),
    )
    save_dir = create_save_directory(tag, best_ckpt)

    save_with_tokenizer(model, save_dir, safe=safe)
    write_modeling_module(
        save_dir,
        template,
        config_class=MISTFinetunedConfig,
        model_class=MISTFinetuned,
        dep_classes=[PredictionTaskHead, AbstractNormalizer, type(model.transform)],
        module_basename="modeling_mist_finetuned",
    )

    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_finetuned",
        model_type="mist_finetuned",
        architecture_name="MISTFinetuned",
        config_class_name="MISTFinetunedConfig",
        model_class_name="MISTFinetuned",
    )

    L.info("Saved finetuned model to %s", save_dir)
    create_tar_gz(save_dir)


def export_conductivity(ckpt: Path) -> MISTIonicConductivity:
    bundle, best_ckpt = load_model(ckpt, model_class="MISTIonicConductivity")

    train_cfg = read_training_config(ckpt)
    tokenizer = load_tokenizer(get_ckpt_tokenizer(best_ckpt))
    n_components = train_cfg.get("n_components") or train_cfg.get("model", {}).get(
        "init_args", {}
    ).get("n_components")
    model = MISTIonicConductivity.from_components(
        encoder=bundle.encoder,
        task_network=bundle.task_network,
        tokenizer=tokenizer,
        n_components=n_components,
    )
    return model, best_ckpt

    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_finetuned",
        model_type="mist_finetuned",
        architecture_name="MISTFinetuned",
        config_class_name="MISTFinetunedConfig",
        model_class_name="MISTFinetuned",
    )

@cli.command()
def conductivity(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """
    Export a mixture ionic-conductivity model.
    """
    model, best_ckpt = export_conductivity(ckpt)

    tag = name_model(
        model,
        template=name or "mist-coductivity-{model_size}-{ckpt}",
        ckpt=ckpt_id(best_ckpt),
    )
    save_dir = create_save_directory(tag, best_ckpt)

    save_with_tokenizer(model, save_dir, safe=safe)
    write_modeling_module(
        save_dir,
        template,
        config_class=MISTIonicConductivityConfig,
        model_class=MISTIonicConductivity,
        dep_classes=[VFTDecayTaskHead],
        module_basename="modeling_mist_ionic_conductivity",
    )
    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_ionic_conductivity",
        model_type="mist_ionic_conductivity",
        architecture_name="MISTIonicConductivity",
        config_class_name="MISTIonicConductivityConfig",
        model_class_name="MISTIonicConductivity",
    )
    L.info("Saved conductivity model to %s", save_dir)
    create_tar_gz(save_dir)


def export_excess_physics(ckpt: Path) -> MISTExcessPhysics:
    bundle, best_ckpt = load_model(ckpt, model_class="MISTExcessPhysics")

    # Training checkpoint - need to construct from components
    train_cfg = read_training_config(ckpt)
    try:
        tokenizer = load_tokenizer(train_cfg["data"]["init_args"]["tokenizer"])
    except KeyError:
        tokenizer = None
    model = MISTExcessPhysics.from_components(
        encoder=bundle.encoder,
        pairwise_interaction=bundle.pairwise_interaction,
        component_properties=bundle.component_properties,
        excess_polynomial=bundle.excess_polynomial,
        transform=bundle.transform,
        excess_transform=bundle.excess_transform,
        tokenizer=tokenizer,
        interactions=train_cfg.get("interactions"),
        num_control=train_cfg.get("num_control"),
        num_targets=len(train_cfg.get("target_columns")),
        temperature_dependence=train_cfg.get("temperature_dependence"),
        relative_excess=train_cfg.get("relative_excess"),
        dropout=train_cfg.get("dropout"),
    )
    return model, best_ckpt


@cli.command()
def excess_physics(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """
    Export an excess physics mixture model.
    """
    model, best_ckpt = export_excess_physics(ckpt)

    tag = name_model(
        model,
        template=name or "mist-excess-{model_size}-{ckpt}",
        ckpt=ckpt_id(best_ckpt),
    )
    save_dir = create_save_directory(tag, best_ckpt)

    save_with_tokenizer(model, save_dir, safe=safe)

    # Get dependency classes for pairwise fusion
    pairwise_class = type(model.pairwise_interaction)
    dep_classes = [
        pairwise_class,
        LagrangePolynomial,
        Standardize,
    ]

    # Add temperature dependence classes if used
    if model.config.temperature_dependence == "arrthenius":
        dep_classes.append(ArrtheniusActivation)
    elif model.config.temperature_dependence == "locally-linear":
        dep_classes.append(LinearExogenousEffect)

    write_modeling_module(
        save_dir,
        template,
        config_class=MISTExcessPhysicsConfig,
        model_class=MISTExcessPhysics,
        dep_classes=dep_classes,
        module_basename="modeling_mist_excess_physics",
    )
    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_excess_physics",
        model_type="mist_excess_physics",
        architecture_name="MISTExcessPhysics",
        config_class_name="MISTExcessPhysicsConfig",
        model_class_name="MISTExcessPhysics",
    )
    L.info("Saved excess physics model to %s", save_dir)
    create_tar_gz(save_dir)


def export_mixtures(ckpt: Path) -> MISTMixtures:
    bundle, best_ckpt = load_model(ckpt, model_class="MISTMixtures")

    train_cfg = read_training_config(ckpt)
    tokenizer = load_tokenizer(get_ckpt_tokenizer(best_ckpt))

    model_cfg = train_cfg.get("model", {}).get("init_args", {})
    n_components = model_cfg.get("n_components", 5)
    output_size = model_cfg.get("output_size", 2)

    temperature_condition = model_cfg.get("temperature", "false")
    if hasattr(temperature_condition, "value"):
        temperature_condition = temperature_condition.value

    # Try to get channels from training config
    try:
        channels = train_cfg["data"]["init_args"].get("target_col")
    except KeyError:
        channels = model_cfg.get("target_columns")

    model = MISTMixtures.from_components(
        encoder=bundle.encoder,
        task_network=bundle.task_network,
        transform=bundle.transform,
        tokenizer=tokenizer,
        n_components=n_components,
        temperature_condition=temperature_condition,
        output_size=output_size,
        channels=channels,
    )
    return model, best_ckpt


@cli.command()
def mixtures(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """
    Export a mixture property prediction model.
    """
    model, best_ckpt = export_mixtures(ckpt)

    tag = name_model(
        model,
        template=name or "mist-mixtures-{model_size}-{ckpt}",
        ckpt=ckpt_id(best_ckpt),
    )
    save_dir = create_save_directory(tag, best_ckpt)

    save_with_tokenizer(model, save_dir, safe=safe)

    # Get dependency classes
    dep_classes = [
        PredictionTaskHead,
        AbstractNormalizer,
        type(model.transform),
        TemperatureCondition,
    ]

    write_modeling_module(
        save_dir,
        template,
        config_class=MISTMixturesConfig,
        model_class=MISTMixtures,
        dep_classes=dep_classes,
        module_basename="modeling_mist_mixtures",
    )

    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_mixtures",
        model_type="mist_mixtures",
        architecture_name="MISTMixtures",
        config_class_name="MISTMixturesConfig",
        model_class_name="MISTMixtures",
    )

    L.info("Saved mixtures model to %s", save_dir)
    create_tar_gz(save_dir)


def export_multitask(encoder_ckpt: Path, task_ckpt: List[Path]) -> MISTMultiTask:
    encoder_ckpt = maybe_best_ckpt(encoder_ckpt)

    try:
        # Try loading from training checkpoints
        encoder = load_encoder(encoder_ckpt)
        tokenizer = get_ckpt_tokenizer(encoder_ckpt)

        task_networks, transforms, channels = [], [], []
        for ckpt in task_ckpt:
            ckpt = maybe_best_ckpt(ckpt)
            cfg = read_training_config(ckpt)
            assert cfg["model"]["init_args"][
                "freeze_encoder"
            ], f"Encoder not frozen for {ckpt}"
            channels.extend(cfg["data"]["init_args"]["target_columns"])
            bundle = SaveConfigWithCkpts.load(ckpt, strict=False)
            task_networks.append(bundle.task_network)
            transforms.append(bundle.transform)

        return MISTMultiTask.from_components(
            encoder=encoder,
            task_networks=task_networks,
            transforms=transforms,
            tokenizer=tokenizer,
            channels=channels,
        )
    except (FileNotFoundError, KeyError):
        L.warning("Could not load from training checkpoints, trying packaged models")

        try:
            encoder_model, _ = load_model(encoder_ckpt)
            encoder = (
                encoder_model.encoder
                if hasattr(encoder_model, "encoder")
                else encoder_model
            )
            tokenizer = (
                encoder_model.tokenizer if hasattr(encoder_model, "tokenizer") else None
            )
        except Exception:
            raise ValueError(f"Could not load encoder from {encoder_ckpt}")

        task_networks, transforms, channels = [], [], []
        for ckpt in task_ckpt:
            ckpt = maybe_best_ckpt(ckpt)
            task_model, _ = load_model(ckpt)

            if isinstance(task_model, MISTFinetuned):
                task_networks.append(task_model.task_network)
                transforms.append(task_model.transform)
                if hasattr(task_model, "channels") and task_model.channels:
                    channels.extend(task_model.channels)
            else:
                raise ValueError(f"Task checkpoint {ckpt} must be a finetuned model")

        return MISTMultiTask.from_components(
            encoder=encoder,
            task_networks=task_networks,
            transforms=transforms,
            tokenizer=tokenizer,
            channels=channels if channels else None,
        )


@cli.command()
def multitask(
    encoder_ckpt: Path,
    task_ckpt: List[Path] = typer.Option(
        [], help="Repeat to add multiple task checkpoints"
    ),
    tasks_in_folder: bool = False,
    name: Optional[str] = None,
    safe: bool = True,
):
    """Export a multitask model."""
    if tasks_in_folder:
        encoder_ckpt = maybe_best_ckpt(encoder_ckpt)
        runs_root = encoder_ckpt.parent.parent.parent
        for d in runs_root.iterdir():
            if d.is_dir() and d.name != "pretrained":
                task_ckpt.append(get_best_ckpt(d))

    model = export_multitask(encoder_ckpt, task_ckpt)

    tag = name_model(
        model,
        template=name or "mist-multi-{model_size}-{ckpt}",
        ckpt=ckpt_id(encoder_ckpt),
    )
    save_dir = create_save_directory(tag, encoder_ckpt, "MISTMultiTask")
    save_with_tokenizer(model, save_dir, safe=safe)

    unique_tf_classes = {type(tf) for tf in model.transforms}
    dep = [PredictionTaskHead, AbstractNormalizer, *unique_tf_classes]

    write_modeling_module(
        save_dir,
        template,
        config_class=MISTMultiTaskConfig,
        model_class=MISTMultiTask,
        dep_classes=dep,
        module_basename="modeling_mist_multitask",
    )

    patch_auto_map(
        save_dir,
        module_basename="modeling_mist_multitask",
        model_type="mist_multitask",
        architecture_name="MISTMultiTask",
        config_class_name="MISTMultiTaskConfig",
        model_class_name="MISTMultiTask",
    )(save_dir)
    L.info("Saved multitask model to %s", save_dir)
    create_tar_gz(save_dir)


if __name__ == "__main__":
    cli()
