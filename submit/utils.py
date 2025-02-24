import json
import logging
import re
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Generator, Optional
from unittest.mock import patch


def dict_product(d: dict[str, list[Any]]) -> Generator[dict[str, Any], None, None]:
    """Yield the product of `d`'s values"""
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def get_ckpt_config(run_id: str) -> dict:
    return json.loads(Path("mist", run_id, "config.json").read_text())


CKPT_REGEX = re.compile(r".*step=(\d+?)-val_loss=([\d\.]+?)\.ckpt")
CKPT_STEP_REGEX = re.compile(r".*step=(\d+?)\.ckpt")


def get_best_ckpt(ckpt_dir) -> str:
    """Return the path for the best checkpoint in a checkpoint directory

    Ties in loss, defers to the checkpoint with more steps
    """
    best_step = None
    best_loss = None
    best = None
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


def get_last_ckpt(path: Path) -> Path:
    """Return the path for the last checkpoint in a checkpoint directory"""
    last_step = None
    last = None

    for ckpt in Path(path, "checkpoints").iterdir():
        if not ckpt.is_dir():
            continue

        if m := CKPT_STEP_REGEX.match(ckpt.name):
            step = int(m.group(1))
        elif m := CKPT_REGEX.match(ckpt.name):
            step = int(m.group(1))
        else:
            continue

        if last is None or step > last_step:
            last = ckpt
            last_step = step

    return str(last.resolve())


def run_id_from_logs(log) -> Optional[str]:
    """Identify the run id from a log file"""
    env_var = re.compile(r"[\+ ]*WANDB_ID=(.+)")
    log_url = re.compile(r"wandb:.*View run at https://wandb.ai/(.*)/runs/(.*)")
    with open(log, "r") as fid:
        for line in fid:
            line = line.strip()
            if m := env_var.match(line):
                return m.group(1)
            if m := log_url.match(line):
                return m.group(2)

    return None


def get_tokenizer(ckpt: str) -> str:
    """Identify tokenizer for a model, potentially remapping `smirk-gpe` paths"""
    model_hparams = json.loads(
        Path(ckpt).parent.parent.joinpath("model_hparams.json").read_text()
    )
    tokenizer = model_hparams["datamodule"]["init_args"]["tokenizer"]

    # Patch in smirk-gpe paths
    tok_path = Path(tokenizer)
    if str(tok_path.name).startswith("smirk-gpe"):
        tokenizer = str(Path(".", tok_path.name).resolve())

    logging.debug("tokenizer: %s -> %s", ckpt, tokenizer)
    return tokenizer


def check_config(config: dict, stage="train"):
    """Validate a run config successfully parses"""

    # This only works for scrips run from the root directory
    # It's a bit of a hack, but seeing as this is for validating configs from
    # one-off sweep scripts, it's okay
    from train import MyLightningCLI, cli_main

    # Override the instantiator to avoid instantiating the model
    def dont_instantiate(self):
        return None

    # Temporarily path MyLightningCLI.instantiate_classes to not instantiate the model
    with patch.object(MyLightningCLI, "instantiate_classes", dont_instantiate):
        with NamedTemporaryFile("w") as f:
            json.dump(config[stage], f)
            f.flush()

            # TODO: Suppress `Seed set to..` message
            cli_main(args=["--config", f.name])
