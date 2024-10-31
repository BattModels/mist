#!/usr/bin/env python
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional
from random import randint

logging.basicConfig(level=logging.INFO)

STATS_DIR = Path(__file__).joinpath("..", "stats").resolve()

REALSPACE = "/lustre/fs0/awadell/realspace_v4_dev"

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

KNOWN_TOKENIZERS = json.loads(Path("tokenizers.json").read_text())

TOKENIZERS = [x["name_or_path"] for x in KNOWN_TOKENIZERS if x["encoding"] == "smile"]

REF_INFO_LOSS = [
    "character",
    "smirk",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-7b",
    "Xenova/gpt-4o",
]


def sbatch(args: list, output: Optional[str] = None, test=True) -> Optional[int]:
    args = [str(x) for x in args]
    if output is not None and Path(output).exists():
        logging.info("skipping job for %s", output)
        return None
    logging.info("submit sbatch: %s => %s", args, output)
    if test:
        return randint(0, 100)

    out = subprocess.run(
        ["/usr/bin/sbatch", *args],
        text=True,
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {out.stderr}")

    return int(out.stdout.split(" ")[-1])


def sub_realspace(tok):
    tok_name = Path(tok).name if Path(tok).is_dir() else tok
    out = STATS_DIR.joinpath(tok_name, "realspace_v4_dev.bson")
    args = [
        "--cpus-per-task=1",
        "--time=1-0:0:0",
        "--ntasks=32",  # Needs to divide datasets evenly (big perf. hit otherwise)
        "submit_tok_stats.sh",
        "usage",
        "--splits=all",
        REALSPACE,
        tok,
    ]
    args = special_case(tok_name, args)

    # id = sbatch(args, out)
    id = None
    return tok_name, id, out


def special_case(tok_name, args):
    if tok_name == "SmilesPE/SPE_ChEMBL":
        args.insert(0, "--mem-per-cpu=4G")
        args.insert(0, "--partition=venkvis-largemem")
    return args


for tok in TOKENIZERS:
    # Submit realspace_v4_dev job
    tok_name, id, realspace_file = sub_realspace(tok)
    ngram_name = realspace_file.with_suffix("").name

    # Compute realspace model loss
    for ds in MOLNET_DATASETS:
        args = [
            "--ntasks=1",
            "--time=4:0:0",
            "--cpus-per-task=1",
            "submit_tok_stats.sh",
            "loss",
            realspace_file,
            ds,
        ]
        if id:
            args.insert(0, f"-d=afterok:{id}")
        args = special_case(tok_name, args)
        if realspace_file.exists():
            sbatch(args, STATS_DIR.joinpath(tok, ds, f"{ngram_name}_model_loss.bson"))

        # Process molnets datsets too
        args = ["--ntasks=1", "--time=1:0:0", "submit_tok_stats.sh", "usage", ds, tok]
        args = special_case(tok_name, args)
        sbatch(args, STATS_DIR.joinpath(tok_name, f"{ds}.bson"))

        # Compute Info loss
        for ref in REF_INFO_LOSS:
            ref_file = STATS_DIR.joinpath(ref, "realspace_v4_dev.bson")
            if not ref_file.exists():
                continue

            args = [
                "--ntasks=48",
                "--cpus-per-task=1",
                "--time=1-0:0:0",
                "submit_tok_stats.sh",
                "distortion",
                f"--reference={ref_file}",
                ds,
                tok,
            ]
            args = special_case(tok_name, args)
            if tok != ref:
                ref_name = ref.replace("/", "--")
                sbatch(
                    args, STATS_DIR.joinpath(tok_name, ds, f"{ref_name}_info_loss.bson")
                )
