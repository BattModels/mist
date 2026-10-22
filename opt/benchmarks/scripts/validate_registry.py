"""Validate the declarative benchmark registry."""

import argparse
from pathlib import Path

from .canonical_tasks import load, load_registry, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--task-definitions", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    canonical = load(args.task_definitions)
    required = {
        "key",
        "dataset",
        "targets",
        "split",
        "metric",
        "unit",
        "table",
        "base",
        "release",
    }
    keys = []
    for case in registry["cases"]:
        case_required = (
            {"key", "base", "release"} if case.get("canonical_task") else required
        )
        if case.get("targets_from"):
            case_required = case_required - {"targets"}
        missing = case_required - case.keys()
        if missing:
            raise ValueError(f"{case.get('key')}: missing {sorted(missing)}")
        if case["base"] not in registry["base_models"]:
            raise ValueError(f"{case['key']}: unknown base")
        resolve(case, canonical)
        if case["release"] is None:
            if "path" not in case:
                raise ValueError(f"{case['key']}: missing dataset path")
            micro = int(case["microbatch"])
            effective = int(case["effective_batch"])
            if micro <= 0 or effective <= 0 or effective % micro:
                raise ValueError(f"{case['key']}: invalid batch contract")
        else:
            for field in ("id", "revision", "validation_value"):
                if field not in case["release"]:
                    raise ValueError(f"{case['key']}: release missing {field}")
            if len(case["release"]["revision"]) != 40:
                raise ValueError(f"{case['key']}: release is not immutable")
        keys.append(case["key"])
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case keys")
    for base in registry["base_models"]:
        for dataset in registry["datasets"]:
            matches = [
                case
                for case in registry["cases"]
                if case.get("canonical_task") == dataset
                and case["base"] == base
                and case.get("target_subset") is None
                and case.get("variant") is None
            ]
            if len(matches) != 1:
                raise ValueError(f"{base}/{dataset}: expected exactly one acquisition")
    expected = {"molnet_classification": 6, "molnet_regression": 5}
    for name, width in expected.items():
        table = registry["paper_tables"][name]
        if len(table["columns"]) != width or len(table["rows"]) != 4:
            raise ValueError(f"{name}: expected four MIST regimes and {width} columns")
        for row in table["rows"]:
            if len(row["values"]) != width or len(row["std"]) != width:
                raise ValueError(f"{name}/{row['model']}: incomplete table row")


if __name__ == "__main__":
    main()
