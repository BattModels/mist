import json
import shutil
from pathlib import Path
from typing import Optional

import typer

from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts, get_ckpt_tokenizer
from electrolyte_fm.utils.tokenizer import load_tokenizer

import utils
from utils import get_best_ckpt, create_save_directory

cli = typer.Typer()


@cli.command()
def pretrained(ckpt: Path, name: Optional[str] = None):
    """Export a pretrained model"""
    if ckpt.joinpath("config.json").is_file():
        ckpt = get_best_ckpt(ckpt)
    model = SaveConfigWithCkpts.load(ckpt)
    name = name or ckpt.parent.parent.name

    save_dir = create_save_directory(name, ckpt)
    model.model.save_pretrained(
        save_directory=save_dir,
        safe_serialization=True,
        push_to_hub=False,
    )
    utils.save_tokenizer(save_dir, ckpt)


def export_finetuned(ckpt: Path):
    from electrolyte_fm.models import MISTFinetuned

    model = SaveConfigWithCkpts.load(ckpt)
    model_config = json.loads(Path(ckpt, "..", "..", "config.json").read_text())
    tokenizer = load_tokenizer(model_config["data"]["init_args"]["tokenizer"])
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
    name = name or ckpt.parent.parent.name
    if Path(ckpt).joinpath("config.json").is_file():
        ckpt = get_best_ckpt(ckpt)
    save_dir = create_save_directory(name, ckpt)
    model = export_finetuned(ckpt)

    utils.export_code(save_dir, model, model.transform, model.task_network)
    utils.save_model(model, save_dir, safe)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))


def export_multitask(
    encoder_ckpt: Path,
    task_ckpt: list[Path],
):
    from electrolyte_fm.models import MISTMultiTask

    encoder = SaveConfigWithCkpts.load(encoder_ckpt).get_encoder()
    tokenizer = get_ckpt_tokenizer(encoder_ckpt)

    task_networks = []
    transforms = []
    channels = []
    for ckpt in task_ckpt:
        print(f"Loading {ckpt}")
        model = SaveConfigWithCkpts.load(ckpt)
        task_networks.append(model.task_network)
        transforms.append(model.transform)

        config = json.loads(Path(ckpt, "..", "..", "config.json").read_text())
        channels.extend(config["data"]["init_args"]["target_columns"])

    return MISTMultiTask(encoder, task_networks, transforms, tokenizer, channels)


@cli.command()
def multitask(
    encoder_ckpt: Path,
    task_ckpt: list[Path] = [],
    tasks_in_folder: bool = False,
    name: Optional[str] = None,
    safe: bool = True,
):
    """Export a Multitask model using a single encoder"""

    name = name or f"{encoder_ckpt.parent.parent.name}-multitask"
    save_dir = create_save_directory(name, encoder_ckpt, "MISTMultiTask")

    # Look in parent folders for task checkpoints
    if tasks_in_folder:
        # If we got a run directory, use the best checkpoint
        if encoder_ckpt.joinpath("config.json").is_file():
            encoder_ckpt = get_best_ckpt(encoder_ckpt)

        for dir in Path(encoder_ckpt).parent.parent.parent.iterdir():
            if dir.is_dir() and dir.name != "pretrained":
                task_ckpt.append(get_best_ckpt(dir))

    model = export_multitask(encoder_ckpt, task_ckpt)
    utils.save_model(model, save_dir, safe)
    utils.export_code(save_dir, model, *model.transforms, *model.task_networks)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))


if __name__ == "__main__":
    cli()
