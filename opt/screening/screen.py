import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch
import typer
from lightning.fabric import Fabric
from src.database import dump_to_sqlite_threaded
from src.generate import OracleCritic, generate

from electrolyte_fm.models.prod_finetune import MISTFinetuned, MISTMultiTask

logging.basicConfig(level=logging.INFO)

HARTREE_TO_EV = 27.211_386_245_981
CAL_TO_JOULES = 4.184

root_dir = Path(__file__).parent.parent.parent

app = typer.Typer()


@app.command()
def main(
    db_name: Optional[str] = None,
    gpus_per_node: Optional[int] = None,
    num_nodes: int = 1,
    precision: str = "bf16",
):
    fabric = Fabric(
        devices=gpus_per_node or "auto",
        num_nodes=num_nodes,
        precision=precision,
    )
    fabric.launch()
    logging.info("rank: %d, world size: %d", fabric.global_rank, fabric.world_size)
    fabric.barrier("startup")

    # Screening critics
    critics = [
        OracleCritic.from_pretrained(
            str(root_dir.joinpath("models/mist-26.9M-b302p09x-bp")),
            limits={"bp": (75, None)},
            model_cls=MISTFinetuned,
        ),
        OracleCritic.from_pretrained(
            str(root_dir.joinpath("models/mist-26.9M-y3ge5pf9-mp")),
            limits={"mp": (None, 0)},
            model_cls=MISTFinetuned,
        ),
        OracleCritic.from_pretrained(
            str(root_dir.joinpath("models/mist-x4i8qzuq-qm9")),
            limits={
                "gap": (5 / HARTREE_TO_EV, None),
                "homo": (None, -7 / HARTREE_TO_EV),
            },
            model_cls=MISTFinetuned,
        ),
    ]

    if db_name is None:
        db_name = fabric.broadcast(str(uuid4()))

    out_dir = Path("out", db_name)
    out_dir.mkdir(exist_ok=True, parents=True)
    dump_to_sqlite_threaded(
        generate(fabric, critics, "./frag_extend.smi.frag", encoding="smiles-kekule"),
        out_dir.joinpath(f"rank_{fabric.global_rank}.sqlite"),
    )


if __name__ == "__main__":
    app()
