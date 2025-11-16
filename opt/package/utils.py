import re
import shutil
import subprocess
import sys
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import jinja2
from smirk import SmirkTokenizerFast

import electrolyte_fm
from electrolyte_fm.utils.ckpt import get_ckpt_tokenizer
from electrolyte_fm.utils.tokenizer import load_tokenizer
from .generate_model_card import generate_model_card_for_directory


def get_best_ckpt(ckpt_dir: Path) -> Path:
    """
    Return the path for the best checkpoint in a checkpoint directory
    Ties in loss, defers to the checkpoint with more steps
    """
    best_step = 0
    best_loss = 0
    best = None
    CKPT_REGEX = re.compile(r".*step=(\d+?)-val_loss=([\d\.]+?)\.ckpt")
    for ckpt in Path(ckpt_dir, "checkpoints").iterdir():
        if m := CKPT_REGEX.match(ckpt.name):
            step = int(m.group(1))
            loss = float(m.group(2))
            update = False
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
    return best.resolve()


def create_save_directory(
    name: str, ckpt: Path, model_class: Optional[str] = None
) -> Path:
    save_dir = Path("hf-models").joinpath(name)
    save_dir.mkdir(exist_ok=True, parents=True)
    write_requirements(save_dir, extra_deps=extra_deps(ckpt))

    license = Path(electrolyte_fm.__file__).parent.joinpath("LICENSE")
    if license.is_file():
        shutil.copy(license, save_dir)

    try:
        model_card = generate_model_card_for_directory(save_dir, ckpt_path=ckpt)
        (save_dir / "README.md").write_text(model_card)
        logging.info(f"Generated model card: {save_dir / 'README.md'}")
    except Exception as e:
        logging.warning(f"Could not generate model card for {save_dir.name}: {e}")
        # Fall back to copying static model card if available
        static_card = Path(__file__).parent.joinpath("model_card.md")
        if static_card.exists():
            shutil.copy(static_card, Path(save_dir, "README.md"))
            logging.info(f"Copied static model card to {save_dir / 'README.md'}")

    if model_class:
        create_demo(save_dir, model_class)

    return save_dir


def create_demo(save_directory: Path, model_class: str):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))
    template = env.get_template("demo_finetune.py.j2")
    script = template.render(model_class=model_class)
    Path(save_directory, "demo.py").write_text(script)


def write_requirements(save_directory: Path, extra_deps: list[str] = []):
    deps = ["transformers", "torch", "scikit-learn", "datasets", *extra_deps]
    with open(save_directory.joinpath("requirements.txt"), "w") as fid:
        for dep in deps:
            # Pin smirk to 0.1.0
            if dep == "smirk":
                fid.write("smirk==0.1.0\n")
            else:
                fid.write(f"{dep}=={version(dep)}\n")


def extra_deps(ckpt: Path) -> list[str]:
    """Determine extra dependencies based on tokenizer type."""
    try:
        tokenizer = load_tokenizer(get_ckpt_tokenizer(ckpt))
    except (ValueError, RuntimeError, FileNotFoundError):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        except Exception:
            # Default to smirk if we can't determine
            return ["smirk"]

    if isinstance(tokenizer, SmirkTokenizerFast):
        return ["smirk"]
    return []


def save_tokenizer(save_dir: Path, ckpt_or_tokenizer):
    if isinstance(ckpt_or_tokenizer, (str, Path)):
        tokenizer = load_tokenizer(str(ckpt_or_tokenizer))
    else:
        tokenizer = ckpt_or_tokenizer

    try:
        tokenizer.save_pretrained(str(save_dir), legacy_format=False, push_to_hub=False)
    except ValueError:
        tokenizer.save_pretrained(str(save_dir))


def export_code(save_dir, *classes):
    files = set()
    for cls in classes:
        mod = cls.__module__
        mod_path = sys.modules[mod].__file__
        assert mod_path is not None
        files.add(Path(mod_path).resolve())

    for file in files:
        dst = shutil.copy(file, save_dir)
        strip_imports(dst)


def save_model(model, save_directory, safe=False):
    # Save the tokenizer first
    if hasattr(model, "tokenizer"):
        save_tokenizer(save_directory, model.tokenizer)

    if safe:
        # For some reason, this avoids an issue when saving the task network
        model.save_pretrained(save_directory, safe_serialization=False)
        model = model.from_pretrained(save_directory)
        Path(save_directory, "model.pt").unlink()

    model.save_pretrained(save_directory, safe_serialization=safe)

    # Validate
    model.__class__.from_pretrained(save_directory)


def strip_imports(file: str, license_header: Optional[str] = None):
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


def metric_units(number):
    unit = ""
    for unit in ["", "K", "M", "B"]:
        if abs(number) < 1000:
            break
        number /= 1000

    return f"{number:.1f}{unit}"


def name_model(model, template: str = "mist-{model_size}", **kwargs):
    kwargs.update(
        {
            "model_size": metric_units(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
        }
    )
    return template.format(**kwargs)


def ckpt_id(path: Path) -> str:
    if path.joinpath("model.safetensors").exists():
        return path.name
    else:
        return path.parent.parent.name


def create_tar_gz(input_dir: Path):
    output = input_dir.parent.joinpath(input_dir.name + ".tar.gz")
    logging.info("Creating %s for %s", output, input_dir)
    subprocess.run(["tar", "-czf", str(output), "-C", str(input_dir), "."], check=True)
