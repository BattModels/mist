"""Run with the synth_access environment; no production model downloads needed."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset

FOLDER = Path(__file__).resolve().parents[1]
ASSEMBLY_FAILURE = "CC(C)(C)OC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"


@pytest.fixture
def workflow(monkeypatch):
    monkeypatch.syspath_prepend(str(FOLDER))
    spec = importlib.util.spec_from_file_location(
        "synth_access_workflow", FOLDER / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "NUM_PROCS", 1)
    return module


def test_unavailable_heuristic_preserves_rows_and_strict_json(workflow):
    data = Dataset.from_dict(
        {
            "smiles": ["CCO", "CCN", "CC", "CCC"],
            "is_hard": [False, True, False, True],
            "row_id": [0, 1, 2, 3],
        }
    )
    result = workflow.evaluate_dataset(
        {"partial": lambda smi: None if smi == "CCO" else float(len(smi))},
        data,
        target="is_hard",
    )
    encoded = json.dumps(result, allow_nan=False)
    rows = json.loads(encoded)["scores"]
    assert rows[0]["partial"] is None
    assert [row["row_id"] for row in rows] == [0, 1, 2, 3]
    assert result["auroc"]["partial"] == 1.0


def test_model_cli_passes_staged_directories_to_evaluation(
    workflow, monkeypatch, tmp_path
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    Dataset.from_dict({"smiles": ["CCO"], "is_hard": [False]}).save_to_disk(
        data_dir / "ba-sascorer"
    )
    output = tmp_path / "scores.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "mist/test",
            str(model_dir),
            "--data-dir",
            str(data_dir),
            "--dataset",
            "ba-sascorer",
            "--output-format",
            str(output),
        ],
    )
    observed = {}

    def evaluate(metrics, dataset, *, target):
        observed.update(metrics=metrics, target=target)
        return {"scores": dataset.to_list()}

    monkeypatch.setattr(workflow, "evaluate_dataset", evaluate)
    workflow.main()

    assert observed == {"metrics": {"mist/test": str(model_dir)}, "target": "is_hard"}
    assert json.loads(output.read_text())["scores"][0]["smiles"] == "CCO"


def test_native_assembly_cli_keeps_known_failure_as_null(tmp_path):
    data = Dataset.from_dict(
        {
            "smiles": [ASSEMBLY_FAILURE, "CCO", "CCN", "c1ccccc1"],
            "is_hard": [True, False, True, False],
        }
    )
    data.save_to_disk(tmp_path / "data/ba-sascorer")
    source_files = {
        path.relative_to(tmp_path / "data"): path.read_bytes()
        for path in (tmp_path / "data").rglob("*")
        if path.is_file()
    }
    output = tmp_path / "scores.json"
    subprocess.run(
        [
            sys.executable,
            str(FOLDER / "main.py"),
            "--data-dir",
            str(tmp_path / "data"),
            "--dataset",
            "ba-sascorer",
            "--metric",
            "assembly-index",
            "--output-format",
            str(output),
        ],
        cwd=FOLDER,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SLURM_CPUS_PER_TASK": "1",
            "OMP_NUM_THREADS": "1",
        },
        check=True,
        timeout=120,
    )

    def reject_non_json(value):
        pytest.fail(f"Non-JSON numeric constant: {value}")

    result = json.loads(output.read_text(), parse_constant=reject_non_json)
    assert len(result["scores"]) == 4
    assert result["scores"][0]["assembly-index"] is None
    assert result["scores"][1]["assembly-index"] == 1
    assert [row["row_id"] for row in result["scores"]] == [0, 1, 2, 3]
    assert 0 <= result["auroc"]["assembly-index"] <= 1
    assert {
        path.relative_to(tmp_path / "data"): path.read_bytes()
        for path in (tmp_path / "data").rglob("*")
        if path.is_file()
    } == source_files
