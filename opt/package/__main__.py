#!/usr/bin/env python
# Script to package MIST models for downstream usage
#
# Usage:
#   python opt/package --help

import sys
import json
import shutil
from pathlib import Path
from typing import Optional
import logging

import typer

sys.path.append(str(Path(__file__).parent.parent.parent))
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts, get_ckpt_tokenizer
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer

import utils
from utils import get_best_ckpt, create_save_directory, ckpt_id

cli = typer.Typer()

logging.basicConfig(level=logging.INFO)


@cli.command()
def pretrained(ckpt: Path, name: Optional[str] = None):
    """Export a pretrained model"""
    if ckpt.joinpath("config.json").is_file():
        ckpt = get_best_ckpt(ckpt)
    model = SaveConfigWithCkpts.load(ckpt)
    name = utils.name_model(
        model,
        template=name or "mist-{model_size}-{ckpt}",
        ckpt=ckpt_id(ckpt),
    )

    save_dir = create_save_directory(name, ckpt)
    model.model.save_pretrained(
        save_directory=save_dir,
        safe_serialization=True,
        push_to_hub=False,
    )
    utils.save_tokenizer(save_dir, ckpt)
    logging.info("Saved model to %s", save_dir)
    utils.create_tar_gz(save_dir)


def export_finetuned(ckpt: Path):
    from electrolyte_fm.models import MISTFinetuned

    model = SaveConfigWithCkpts.load(ckpt)
    model_config = json.loads(ckpt.parent.parent.joinpath("config.json").read_text())
    tokenizer_name = model_config["data"]["init_args"]["tokenizer"]
    tokenizer = load_tokenizer(tokenizer_name)
    return MISTFinetuned(
        model.encoder,
        model.task_network,
        model.transform,
        tokenizer=tokenizer,
        channels=model_config["data"]["init_args"]["target_columns"],
    )


@cli.command()
def finetuned(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """Export a finetuned model"""
    if Path(ckpt).joinpath("config.json").is_file():
        ckpt = get_best_ckpt(ckpt)
    model = export_finetuned(ckpt)

    name = utils.name_model(
        model,
        template=name or "mist-{model_size}-{ckpt}",
        ckpt=ckpt_id(ckpt),
    )
    save_dir = create_save_directory(name, ckpt)
    utils.export_code(save_dir, model, model.transform, model.task_network)
    utils.save_model(model, save_dir, safe)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))
    logging.info("Saved model to %s", save_dir)
    utils.create_tar_gz(save_dir)


@cli.command()
def mixtures(ckpt: Path, name: Optional[str] = None, safe: bool = True):
    """Export a mixture model"""
    if Path(ckpt).joinpath("config.json").is_file():
        ckpt = get_best_ckpt(ckpt)

    model = DeepSpeedMixin.load(ckpt)

    name = utils.name_model(
        model,
        template=name or "mist-{model_size}-{ckpt}",
        ckpt=ckpt_id(ckpt),
    )
    save_dir = create_save_directory(name, ckpt)
    utils.export_code(save_dir, model, model.transform, model.task_network)
    utils.save_model(model, save_dir, safe)
    logging.info("Saved model to %s", save_dir)
    utils.create_tar_gz(save_dir)


def export_multitask(
    encoder_ckpt: Path,
    task_ckpt: list[Path],
):
    from electrolyte_fm.models import MISTMultiTask
    from electrolyte_fm.models.lm_finetuning import load_encoder

    encoder = load_encoder(encoder_ckpt)
    tokenizer = get_ckpt_tokenizer(encoder_ckpt)

    task_networks = []
    transforms = []
    channels = []
    for ckpt in task_ckpt:
        logging.info(f"Loading {ckpt}")
        config = json.loads(Path(ckpt, "..", "..", "config.json").read_text())
        channels.extend(config["data"]["init_args"]["target_columns"])

        frozen = config["model"]["init_args"]["freeze_encoder"]
        assert frozen, f"Encoder was not frozen for {ckpt}"

        model = SaveConfigWithCkpts.load(ckpt)
        task_networks.append(model.task_network)
        transforms.append(model.transform)

    return MISTMultiTask(encoder, task_networks, transforms, tokenizer, channels)


@cli.command()
def multitask(
    encoder_ckpt: Path,
    task_ckpt: list[Path] = [],
    tasks_in_folder: bool = False,
    name: Optional[str] = None,
    safe: bool = True,
):
    """Export a Multitask model using a single encoder

    Multiple task networks can be specified by repeating `--task-ckpt`

    > All task checkpoints must be frozen!
    """

    # Look in parent folders for task checkpoints
    if tasks_in_folder:
        # If we got a run directory, use the best checkpoint
        if encoder_ckpt.joinpath("config.json").is_file():
            encoder_ckpt = get_best_ckpt(encoder_ckpt)

        for dir in Path(encoder_ckpt).parent.parent.parent.iterdir():
            if dir.is_dir() and dir.name != "pretrained":
                task_ckpt.append(get_best_ckpt(dir))

    model = export_multitask(encoder_ckpt, task_ckpt)
    name = utils.name_model(
        model,
        template=name or "mist-{model_size}-{encoder}-multitask",
        encoder=ckpt_id(encoder_ckpt),
    )
    save_dir = create_save_directory(name, encoder_ckpt, "MISTMultiTask")
    utils.save_model(model, save_dir, safe)
    utils.export_code(save_dir, model, *model.transforms, *model.task_networks)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))
    logging.info("Saved model to %s", save_dir)
    utils.create_tar_gz(save_dir)


if __name__ == "__main__":
    cli()
