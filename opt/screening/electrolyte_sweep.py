#!/usr/bin/env -S uv run python
import argparse
import json
from pathlib import Path

from electrolyte_fm.design import HyperSpace

sweep = {
    "gpus": [1, 2, 4, 8],
    "batch_size": [128, 256, 512, 1024],
    "n_fragments": [4000, 8000],
    "epoch_fraction": [0.9, 1, 1.1],
    "limit_db_fragments": [None, 1 / 2, 1 / 4, 0.1],
}


def main():
    parser = argparse.ArgumentParser(description="Generate screening sweep parameters")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON sweep configuration"
    )
    args = parser.parse_args()
    cases = sorted(HyperSpace(sweep).gsd(r=4), key=lambda d: d["gpus"])
    runs = []
    for case in cases:
        settings = dict(case)
        settings["epoch_size"] = int(
            settings["n_fragments"] * settings.pop("epoch_fraction")
        )
        settings["seconds"] = 300
        runs.append(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
