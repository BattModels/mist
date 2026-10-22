"""Load benchmark registries and canonical MoleculeNet task definitions."""

import json
from pathlib import Path

import _jsonnet
import yaml


def load(path: Path) -> dict:
    return json.loads(_jsonnet.evaluate_file(str(path.resolve())))


def load_registry(path: Path) -> dict:
    """Anchor target-schema references to the registry, independent of cwd."""
    path = path.resolve()
    registry = yaml.safe_load(path.read_text())
    for case in registry["cases"]:
        if "targets_from" in case:
            case["targets_from"] = str((path.parent / case["targets_from"]).resolve())
    return registry


def resolve(case: dict, canonical: dict) -> dict:
    task_id = case.get("canonical_task")
    if task_id is None:
        if "targets_from" not in case:
            return case
        schema = yaml.safe_load(Path(case["targets_from"]).read_text())
        return {**case, "targets": [item["name"] for item in schema["channels"]]}
    if task_id not in canonical:
        raise ValueError(f"unknown canonical task: {task_id}")
    task = canonical[task_id]
    targets = case.get("target_subset", task["target_columns"])
    if not set(targets).issubset(task["target_columns"]):
        raise ValueError(f"{case['key']}: target subset is outside {task_id}")
    return {
        **case,
        "dataset": task_id,
        "targets": targets,
        "split": task["split"],
        "task": task["task"],
        "metrics": task["metrics"],
    }
