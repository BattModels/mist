"""Changing baseline code compatibility changes the staged model artifact."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("stage_key", [None, "surprise-molformer-transformers-v4"])
def test_stage_key_selects_task_and_output_identity(stage_key):
    source = Path(__file__).resolve().parents[2] / "benchmarks/workflow.pomset"
    definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "stage_base"
    )
    namespace = {
        "Path": Path,
        "ROOT": Path("/results/benchmarks"),
        "ENV": {},
        "task": lambda **kwargs: kwargs,
        "shell": lambda: "shell",
        "validate_checksums": lambda **kwargs: "checksums",
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    base = {"id": "ibm-research/MoLFormer-XL-both-10pct", "revision": "pinned"}
    if stage_key is not None:
        base["stage_key"] = stage_key
    task = namespace["stage_base"](
        "surprise-molformer",
        SimpleNamespace(outputs=SimpleNamespace(tasks="canonical")),
        SimpleNamespace(outputs=SimpleNamespace(python="python")),
        base=base,
    )
    identity = stage_key or "surprise-molformer"
    assert task["name"] == "benchmark-stage-" + identity
    assert task["output_dir"] == Path("/results/benchmarks/bases") / identity
    assert task["cmd"][task["cmd"].index("--revision") + 1] == "pinned"
    assert task["cmd"][task["cmd"].index("--output") + 1] == (
        task["output_dir"] / "model"
    )
    assert task["outputs"] == {"model": Path("model")}
