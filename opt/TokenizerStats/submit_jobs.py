#!/usr/bin/env python
import logging
import subprocess
from pathlib import Path
from typing import Optional
from random import randint

from electrolyte_fm.data_modules.molnet_dataset import _URLS as MOLNET_URLS

logging.basicConfig(level=logging.INFO)

STATS_DIR = Path(__file__).joinpath("..", "stats").resolve()

REALSPACE = "/nfs/turbo/coe-venkvis/mist/realspace_v4_dev"

TOKENIZERS = [
    "smirk",
    "character",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "SmilesPE/SPE_ChEMBL",
    "devalab/molgpt-moses",
    "devalab/molgpt-guacamol",
    "MolecularAI/Chemformer",
    "MolecularAI/Chemformer-downstream",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
    "sagawa/ReactionT5-yield-prediction",
    "rxn4chemistry/rxn_yields",
    "rxn4chemistry/rxnfp",
    "ChangwenXu98/TransPolymer",
    "../../smirk-gpe-50k-mb-ss",
    "../../smirk-gpe-50k-nmb-ss",
    "../../smirk-gpe-small-50k-mb-ss",
    "google/gemma-7b",
    "Xenova/gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3-8B",
]


def sbatch(args: list, output: Optional[str] = None, test=False) -> Optional[int]:
    args = [str(x) for x in args]
    if output is not None and Path(output).exists():
        # logging.info("skipping job for %s", output)
        return None
    logging.info("submit sbatch: %s", args)
    if test:
        return randint(0, 100)

    out = subprocess.run(
        args,
        executable="sbatch",
        text=True,
        capture_output=True,
        check=True,
    )
    return int(out.stdout.split(" ")[-1])


def sub_realspace(tok):
    tok_name = Path(tok).parent.name if Path(tok).is_dir() else tok
    out = STATS_DIR.joinpath(tok_name, "realspace_v4_dev.bson")
    id = sbatch(
        [
            "--time=1-0:0:0",
            "--ntasks=64",
            "submit_tok_stats.sh",
            "usage",
            REALSPACE,
            tok,
        ],
        out,
    )
    return tok_name, id, out


_, char_id, char_realspace = sub_realspace("character")
assert char_id is not None


for tok in TOKENIZERS:
    # Submit realspace_v4_dev job
    if tok != "character":
        tok_name, id, file = sub_realspace(tok)
    else:
        tok_name = "character"
        id = char_id
        file = char_realspace

    # Compute realspace model loss
    for ds in MOLNET_URLS.keys():
        args = ["--ntasks=1", "submit_tok_stats.sh", "loss", file, ds]
        if id:
            args.insert(0, f"-d=afterok:{id}")
        sbatch(args, STATS_DIR.joinpath(tok, f"{ds}_model_loss.bson"))

        # Process molnets datsets too
        sbatch(
            ["-n=1", "submit_tok_stats.sh", "usage", ds, tok],
            STATS_DIR.joinpath(tok_name, f"{ds}.bson"),
        )

        # Compute Info loss
        args = [
            "--ntasks=32",
            "submit_tok_stats.sh",
            "distortion",
            f"--reference={char_realspace}",
            ds,
            tok,
        ]
        if char_id:
            args.insert(0, f"-d=afterok:{char_id}")

        if tok != "character":
            sbatch(args, STATS_DIR.joinpath(f"{ds}_character_info_loss.bson"))
