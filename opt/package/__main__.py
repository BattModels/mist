import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import jinja2
import typer

import electrolyte_fm
from electrolyte_fm.models import MISTFinetuned, MISTMultiTask
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts, get_ckpt_tokenizer
from electrolyte_fm.utils.tokenizer import load_tokenizer

cli = typer.Typer()


def create_save_directory(name: str, ckpt: str) -> Path:
    save_dir = Path(name)
    save_dir.mkdir(exist_ok=True, parents=True)
    write_requirements(save_dir, extra_deps=extra_deps(ckpt))

    license = Path(electrolyte_fm.__file__).parent.joinpath("LICENSE")
    if license.is_file():
        shutil.copy(license, save_dir)

    shutil.copy(
        Path(__file__).parent.joinpath("model_card.md"),
        Path(save_dir, "README.md"),
    )

    return save_dir


def create_demo(save_directory: Path, model_class: str):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))
    template = env.get_template("demo_finetune.py.j2")
    script = template.render(model_class=model_class)
    Path(save_directory, "demo.py").write_text(script)


def write_requirements(save_directory: Path, extra_deps: list[str] = []):
    deps = ["transformers", "torch", *extra_deps]
    with open(save_directory.joinpath("requirements.txt"), "w") as fid:
        for dep in deps:
            fid.write(f"{dep}=={version(dep)}\n")


def extra_deps(ckpt: str) -> list[str]:
    tokenizer = get_ckpt_tokenizer(ckpt)
    if "smirk" in tokenizer:
        return ["smirk"]


def save_tokenizer(save_dir: Path, ckpt: str):
    tokenizer = load_tokenizer(str(ckpt))
    try:
        tokenizer.save_pretrained(save_dir, legacy_format=False, push_to_hub=False)
    except:
        tokenizer.save_pretrained(save_dir)


def export_code(save_dir, *classes):
    files = set()
    for cls in classes:
        mod = cls.__module__
        files.add(Path(sys.modules[mod].__file__).resolve())

    for file in files:
        dst = shutil.copy(file, save_dir)
        strip_imports(dst)


def save_model(model, save_directory, safe=False):
    if safe:
        # For some reason, this avoids an issue when saving the task network
        model.save_pretrained(save_directory, safe_serialization=False)
        model = model.from_pretrained(save_directory)
        Path(save_directory, "model.pt").unlink()

    model.save_pretrained(save_directory, safe_serialization=safe)

    # Validate
    model.__class__.from_pretrained(save_directory)


def strip_imports(file: str, license_header: str = None):
    with open(file, "a") as fid:
        p = subprocess.run(
            ["sed", r"s/from electrolyte_fm\.models\./from /", str(file)],
            check=True,
            text=True,
            capture_output=True,
        )
    with open(file, "w") as fid:
        if license_header is not None:
            fid.write(license_header)
        fid.write(p.stdout)


def get_best_ckpt(ckpt_dir) -> str:
    """Return the path for the best checkpoint in a checkpoint directory

    Ties in loss, defers to the checkpoint with more steps
    """
    best_step = None
    best_loss = None
    best = None
    CKPT_REGEX = re.compile(r".*step=(\d+?)-val_loss=([\d\.]+?)\.ckpt")
    for ckpt in Path(ckpt_dir, "checkpoints").iterdir():
        if not ckpt.is_dir():
            continue
        if m := CKPT_REGEX.match(ckpt.name):
            step = int(m.group(1))
            loss = float(m.group(2))
            if best is None:
                update = True
            elif loss < best_loss:
                update = True
            elif loss <= best_loss and step > best_step:
                update = True

            if update:
                update = False
                best = ckpt
                best_step = step
                best_loss = loss

    assert best is not None
    return str(best.resolve())


@cli.command()
def pretrained(ckpt: str, name: Optional[str] = None):
    """Export a pretrained model"""
    model = SaveConfigWithCkpts.load(ckpt)
    name = name or ckpt.parent.parent.name

    save_dir = create_save_directory(name, ckpt)
    model.model.save_pretrained(
        save_directory=save_dir,
        safe_serialization=True,
        push_to_hub=False,
    )
    save_tokenizer(save_dir, ckpt)


def export_finetuned(ckpt: str):
    model = SaveConfigWithCkpts.load(ckpt)
    model_config = json.loads(Path(ckpt, "..", "..", "config.json").read_text())
    return MISTFinetuned(
        model.encoder,
        model.task_network,
        model.transform,
        channels=model_config["data"]["init_args"]["target_columns"],
    )


@cli.command()
def finetuned(ckpt: str, name: Optional[str] = None, safe: bool = True):
    """Export a finetuned model"""
    name = name or Path(ckpt).parent.parent.name
    save_dir = create_save_directory(name, ckpt)
    model = export_finetuned(ckpt)

    export_code(save_dir, model, model.transform, model.task_network)
    save_model(model, save_dir, safe)
    save_tokenizer(save_dir, ckpt)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))


def export_multitask(
    encoder_ckpt: str,
    task_ckpt: list[str],
):
    encoder = SaveConfigWithCkpts.load(encoder_ckpt).get_encoder()

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

    return MISTMultiTask(encoder, task_networks, transforms, channels)


@cli.command()
def multitask(
    encoder_ckpt: str,
    task_ckpt: list[str] = [],
    tasks_in_folder: bool = False,
    name: Optional[str] = None,
    safe: bool = True,
):
    """Export a Multitask model using a single encoder"""

    encoder_ckpt = Path(encoder_ckpt)
    name = name or f"{encoder_ckpt.parent.parent.name}-multitask"
    save_dir = create_save_directory(name, encoder_ckpt)
    create_demo(save_dir, "MISTMultiTask")

    # Look in parent folders for task checkpoints
    if tasks_in_folder:
        # If we got a run directory, use the best checkpoint
        if encoder_ckpt.joinpath("config.json").is_file():
            encoder_ckpt = get_best_ckpt(encoder_ckpt)

        for dir in Path(encoder_ckpt).parent.parent.parent.iterdir():
            if dir.is_dir() and dir.name != "pretrained":
                task_ckpt.append(get_best_ckpt(dir))

    model = export_multitask(encoder_ckpt, task_ckpt)
    save_model(model, save_dir, safe)
    save_tokenizer(save_dir, encoder_ckpt)
    export_code(save_dir, model, *model.transforms, *model.task_networks)
    shutil.move(Path(save_dir, "prod_finetune.py"), Path(save_dir, "model.py"))


if __name__ == "__main__":
    cli()
