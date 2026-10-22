"""Validate a pre-materialized replacement split and publish its Pomset outputs."""

import argparse
from pathlib import Path

import pandas as pd

from .canonical_tasks import load, load_registry, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task-definitions", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    raw_case = next(case for case in registry["cases"] if case["key"] == args.case)
    targets = resolve(raw_case, load(args.task_definitions))["targets"]
    for split in ("train", "validation", "test"):
        path = args.dataset / f"{split}.csv"
        frame = pd.read_csv(path)
        required = {"smiles", *targets}
        missing = required - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f"{path}: missing={sorted(missing)}, rows={len(frame)}")


if __name__ == "__main__":
    main()
