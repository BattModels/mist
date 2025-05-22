import json
import sys
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4
from itertools import islice

import typer
import yaml
from lightning.fabric import Fabric
from src.database import dump_to_sqlite_threaded
from src.dataloader import DatabaseFragmentDataset
from src.generate import OracleCritic, generate
from src.utils import configure_logging, take_for_seconds
from torch.autograd.profiler import emit_nvtx

from electrolyte_fm.models.prod_finetune import MISTFinetuned

HARTREE_TO_EV = 27.211_386_245_981
CAL_TO_JOULES = 4.184

root_dir = Path(__file__).parent.parent.parent

app = typer.Typer()


def load_config(path: Path):
    if path == Path("-"):
        input_text = sys.stdin.read()
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            return yaml.safe_load(input_text)
    elif path.suffix in [".yaml", ".yml"]:
        with path.open("r") as f:
            return yaml.safe_load(f)
    else:
        with path.open("r") as f:
            return json.load(f)


@app.command()
def main(
    config_path: Path = Path("config.yaml"),
    run_name: Optional[str] = None,
    gpus_per_node: Optional[int] = None,
    num_nodes: int = 1,
    precision: str = "bf16",
    out_dir: Path = Path("out"),
):
    fabric = Fabric(
        devices=gpus_per_node or "auto",
        num_nodes=num_nodes,
        precision=precision,
    )
    fabric.launch()
    logging.info("rank: %d, world size: %d", fabric.global_rank, fabric.world_size)
    fabric.barrier("startup")

    # Configure output directory
    if run_name is None:
        run_name = fabric.broadcast(str(uuid4()))

    out_dir = out_dir.joinpath(run_name)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Configure logging
    configure_logging(fabric, out_dir.joinpath("screen.jsonl"))

    # Load critic config
    config = load_config(config_path)
    json_config_path = Path(out_dir, "config.json")
    if fabric.global_rank == 0:
        with open(json_config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Screening critics
    critics = [
        OracleCritic.from_pretrained(
            str(root_dir / critic["model_path"]),
            limits=critic["limits"],
            model_cls=eval(critic["model_cls"]),
        )
        for critic in config["critics"]
    ]

    # Fragment dataset
    mol_generator = DatabaseFragmentDataset(
        fabric,
        critics[0].oracle.tokenizer,
        db_path=config["generation"]["db_path"],
        n_fragments=config["generation"]["n_fragments"],
        ref_frag_file=config["generation"]["reference_fragments"],
        epoch_size=config["generation"]["epoch_size"],
        limit_db_fragments=config["generation"]["limit_db_fragments"],
        limit_ref_fragments=config["generation"]["limit_ref_fragments"],
    ).dataloader(config["generation"]["batch_size"])

    if (limit := config.get("limit", None)) is not None:
        mol_generator = islice(mol_generator, limit)

    if (limit := config.get("limit_walltime", None)) is not None:
        mol_generator = take_for_seconds(mol_generator, limit)

    with emit_nvtx():
        dump_to_sqlite_threaded(
            generate(
                fabric,
                critics,
                mol_generator,
            ),
            out_dir.joinpath(f"rank_{fabric.global_rank}.sqlite"),
        )


if __name__ == "__main__":
    app()
