"""Expose a validation-selected sweep model as a portable directory artifact."""

import argparse
import json
from pathlib import Path

from electrolyte_fm.utils.stage_hf import link_model

from .checkpoint_selection import select_lr


def select(cells, models, output):
    if len(cells) != len(models):
        raise ValueError("Each sweep cell requires its model directory")
    records = [json.loads(path.read_text()) for path in cells]
    index = select_lr(records)
    output.mkdir(parents=True, exist_ok=True)
    link_model(models[index], output / "model", link_files=True)
    (output / "selection.json").write_text(json.dumps({
        "selection_split": "validation",
        "test_data_used": False,
        "selected": records[index],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select(args.cells, args.models, args.output_dir)


if __name__ == "__main__":
    main()
