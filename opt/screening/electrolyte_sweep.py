#!/usr/bin/env -S uv run python
import yaml
from copy import deepcopy
import json
from pathlib import Path
from itertools import product
from subprocess import run
from uuid import uuid4

from electrolyte_fm.design import HyperSpace


def dict_product(d: dict[str, list]):
    """Yield the product of `d`'s values"""
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


sweep = {
    "gpus": [1, 2, 4, 8],
    "batch_size": [128, 256, 512, 1024],
    "n_fragments": [100, 500, 1000, 2000],
    "epoch_fraction": [0.9, 1, 1.1],
    "limit_ref_fragments": [None, 1 / 2, 1 / 4, 0.1],
    "limit_db_fragments": [None, 1 / 2, 1 / 4, 0.1],
}

nunique_db_fragments = 88_800_000
nunique_ref_fragments = 122

sweep_dir = "initial-sweep"

ref = yaml.safe_load(Path("config.yaml").read_text())
hs = HyperSpace(sweep).gsd(r=4)
for c in sorted(hs, key=lambda d: d["gpus"]):
    print(c)
    config = deepcopy(ref)
    config["generation"].update(
        {
            "batch_size": c["batch_size"],
            "n_fragments": c["n_fragments"],
            "epoch_size": int(c["n_fragments"] * c["epoch_fraction"]),
            "limit_db_fragments": int(c["limit_db_fragments"] * nunique_db_fragments)
            if c["limit_db_fragments"] is not None
            else None,
            "limit_ref_fragments": int(c["limit_ref_fragments"] * nunique_ref_fragments)
            if c["limit_ref_fragments"] is not None
            else None,
        }
    )

    run_id = str(uuid4())
    out_dir = Path(sweep_dir, run_id)
    out_dir.mkdir(exist_ok=True, parents=True)
    with Path(sweep_dir, run_id, "config.json").open("w") as f:
        json.dump(config, f)

    run(
        [
            "sbatch",
            "--nodes=1",
            f"--ntasks-per-node={c['gpus']}",
            f"--gpus-per-node={c['gpus']}",
            f"--output={out_dir}/slurm-%A_%a.out",
            "launch.sh",
            "--out-dir",
            Path(sweep_dir).resolve(),
            "--run-name",
            run_id,
            "--config-path",
            Path(sweep_dir, run_id, "config.json").resolve(),
        ],
        text=True,
        check=True,
    )
