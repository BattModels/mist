#!/usr/bin/env python
import logging
import subprocess
from pathlib import Path
from typing import Optional
from random import randint

logging.basicConfig(level=logging.INFO)

STATS_DIR = Path(__file__).joinpath("..", "stats").resolve()

REALSPACE = "/nfs/turbo/coe-venkvis/mist/realspace_v4_dev2"

MOLNET_DATASETS = [
    "qm8",
    "qm9",
    "esol",
    "freesolv",
    "lipo",
    "muv",
    "hiv",
    "bace",
    "bbbp",
    "tox21",
    "toxcast",
    "sider",
    "clintox",
]

TOKENIZERS = [
    "character",
    "smirk",
    "ibm/MoLFormer-XL-both-10pct-oov",
    # "SmilesPE/SPE_ChEMBL",
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
    "./smirk-gpe-50k-mb-ss",
    "./smirk-gpe-50k-nmb-ss",
    "./smirk-gpe-small-50k-mb-ss",
    "google/gemma-7b",
    "google/gemma-2-2b",
    "Xenova/gpt-4o",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
]

REF_INFO_LOSS = [
    "character",
    "meta-llama/Meta-Llama-3.1-8B",
    # "google/gemma-7b",
    # "Xenova/gpt-4o",
]

PENDING_JOBS = dict()


def sbatch(
    args: dict,
    output: Optional[str] = None,
    deps: Optional[list] = None,
    test=False,
) -> Optional[str]:
    if output is not None and Path(output).exists():
        logging.info("skipping job for %s", output)
        return None

    # Add dependencies
    job_dependencies = set(args.get("dependency", []))
    for dep in deps or []:
        if dep in PENDING_JOBS:
            job_dependencies.add(f"afterok:{PENDING_JOBS[dep]}")
    args["dependency"] = list(job_dependencies)

    cmd = []
    for k, v in args.items():
        if k in ["cmd", "args"]:
            continue
        if isinstance(v, list):
            cmd.extend([f"--{k}={str(x)}" for x in v])
        else:
            cmd.append(f"--{k}={str(v)}")

    cmd.append(args["cmd"])
    if job_args := args.get("args", None):
        cmd.extend([str(x) for x in job_args])
    logging.info("submit sbatch: %s => %s", cmd, output)

    # Fake id for test mode
    if test:
        id = randint(0, 100)
    else:
        out = subprocess.run(
            ["/usr/bin/sbatch", *cmd],
            text=True,
            capture_output=True,
        )
        if out.returncode != 0:
            raise RuntimeError(f"sbatch failed: {out.stderr}")
        id = int(out.stdout.split(" ")[-1])

    # Record pending job
    if output:
        PENDING_JOBS[output] = id
    return id


def sub_realspace(tok):
    tok_name = Path(tok).name if Path(tok).is_dir() else tok
    out = STATS_DIR.joinpath(tok_name, "realspace_v4_dev2.bson")
    if tok_name == "SmilesPE/SPE_ChEMBL":
        out = out.with_suffix(".jld")
    args = {
        "cpus-per-task": 1,
        "time": "8:0:0",
        "cmd": "submit_tok_stats.sh",
        "args": ["usage", "--splits=all", REALSPACE, tok],
    }
    args = special_case(tok_name, args)
    id = sbatch(args, out)
    return tok_name, id, out


def special_case(tok_name, args: dict) -> dict:
    if tok_name == "SmilesPE/SPE_ChEMBL":
        args["mem-per-cpu"] = "4G"
        args["partition"] = "venkvis-largemem"
        args["cpus-per-task"] = 3

    return args


for tok in TOKENIZERS:
    # Submit realspace_v4_dev job
    tok_name, id, realspace_file = sub_realspace(tok)
    ngram_name = realspace_file.with_suffix("").name

    # Model Loss for realspace_v4_dev
    args = {
        "ntasks": 64,
        "cpus-per-task": 1,
        "time": "8:0:0",
        "cmd": "submit_tok_stats.sh",
        "args": ["loss", realspace_file, tok, REALSPACE],
    }
    args = special_case(tok_name, args)
    sbatch(
        args,
        STATS_DIR.joinpath(tok, "realspace_v4_dev2", f"{ngram_name}_model_loss.bson"),
        deps=[realspace_file],
    )

    # Compute realspace model loss
    for ds in MOLNET_DATASETS:
        args = {
            "ntasks": 1,
            "time": "4:0:0",
            "cmd": "submit_tok_stats.sh",
            "args": ["loss", realspace_file, tok, ds],
        }
        args = special_case(tok_name, args)
        sbatch(
            args,
            STATS_DIR.joinpath(tok_name, ds, f"{ngram_name}_model_loss.bson"),
            deps=[realspace_file],
        )

        # Process molnets datsets too
        args = {
            "ntasks": 1,
            "time": "1:0:0",
            "cmd": "submit_tok_stats.sh",
            "args": ["usage", "--splits=all", ds, tok],
        }
        args = special_case(tok_name, args)
        model = STATS_DIR.joinpath(tok_name, f"{ds}.bson")
        sbatch(args, model)

        # Plus loss for downsteam models
        args = {
            "ntasks": 1,
            "time": "0:30:0",
            "cmd": "submit_tok_stats.sh",
            "args": ["loss", model, tok, ds],
        }
        args = special_case(tok_name, args)
        sbatch(
            args,
            STATS_DIR.joinpath(tok_name, ds, f"{ds}_model_loss.bson"),
            deps=[model],
        )

for tok in TOKENIZERS:
    for ds in MOLNET_DATASETS:
        # Compute Info loss
        for ref in REF_INFO_LOSS:
            if tok == ref:
                continue
            ref_file = STATS_DIR.joinpath(ref, "realspace_v4_dev2.bson")
            asgs = {
                "ntasks": 24,
                "cpus-per-task": 2,
                "time": "0:30:0",
                "cmd": "submit_tok_stats.sh",
                "args": ["distortion", f"--reference={ref_file}", ds, tok],
            }
            args = special_case(tok_name, args)
            ref_name = ref.replace("/", "--")
            sbatch(
                args,
                STATS_DIR.joinpath(tok_name, ds, f"{ref_name}_info_loss.bson"),
                deps=[ref_file],
            )
