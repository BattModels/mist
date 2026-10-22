"""Exercise the real registry's complete two-family report with synthetic scores."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "opt/benchmarks/registry.yaml"


@pytest.fixture
def results(tmp_path):
    registry = yaml.safe_load(REGISTRY.read_text())
    records = []
    for case in registry["cases"]:
        if case.get("target_subset"):
            continue
        row = {
            "case": case["key"],
            "base": case["base"],
            "dataset": case.get("canonical_task", case.get("dataset")),
            "variant": case.get("variant"),
            "split": "test",
            "selected_by": "validation only",
            "provenance": "synthetic test fixture",
            "metrics": {"mae": 0.1, "rmse": 0.2},
        }
        if row["dataset"] in ("bbbp", "muv", "odour"):
            row["metrics"] = {"auroc": 0.8, "avg-precision": 0.7}
        path = tmp_path / f"{case['key']}.json"
        path.write_text(json.dumps(row))
        records.append(path)
    return records


def run_report(results, output):
    return subprocess.run(
        [
            sys.executable,
            "-m", "opt.benchmarks.scripts.collate_validation",
            "--registry",
            str(REGISTRY),
            "--tests",
            *map(str, results),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_report_contains_all_bases_datasets_variants_and_distinct_references(
    tmp_path, results
):
    output = tmp_path / "report"
    result = run_report(results, output)
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "validation-benchmarks.json").read_text())
    records = payload["held_out_tests"]
    assert len(records) == 46
    assert len({row["case"] for row in records}) == len(records)
    assert {row["base"] for row in records} == {"mist-28m", "mist-1p8b"}
    assert len([row for row in records if row["dataset"] == "pubchemqc"]) == 2
    assert len([row for row in records if row["dataset"] == "isotopes"]) == 2
    assert len([row for row in records if row.get("variant")]) == 2
    assert payload["selection_test_used"] is False
    assert payload["held_out_test_scored"] is True
    with (output / "benchmark-tests.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert {row["case"] for row in csv_rows} == {row["case"] for row in records}
    latex = (output / "benchmark-tests.tex").read_text()
    assert "qm9 (baseline)" in latex and "qm9 (randomized)" in latex
    assert len(latex.splitlines()) == 53
    # The old paper's reference values must never be labelled computed scores.
    for name, table in payload["paper_tables"].items():
        assert (output / f"{name}.tex").is_file()
        assert (output / f"{name}.csv").is_file()
        assert {row["provenance"] for row in table["rows"]} == {
            "paper_reported_validation"
        }


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "wrong_base", "wrong_variant", "test_selection"]
)
def test_report_rejects_incomplete_or_misattributed_results(tmp_path, results, fault):
    if fault == "missing":
        results.pop()
    elif fault == "duplicate":
        results.append(results[0])
    else:
        path = results[0]
        row = json.loads(path.read_text())
        field, value = {
            "wrong_base": ("base", "another-model"),
            "wrong_variant": ("variant", "randomized"),
            "test_selection": ("selected_by", "best test loss"),
        }[fault]
        row[field] = value
        path.write_text(json.dumps(row))
    output = tmp_path / "report"
    result = run_report(results, output)
    assert result.returncode != 0
    assert not (output / "validation-benchmarks.json").exists()
