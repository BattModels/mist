"""Lock established split membership and row order; never replace completed data."""

import json
import socket
from pathlib import Path

import pandas as pd
import pytest
import yaml
from datasets import Dataset

from electrolyte_fm.data_modules.utils import train_val_test_split
from opt.benchmarks.scripts.materialize_application import (
    ESTABLISHED_TASKS,
    read_cached_source,
    read_source,
    write_splits,
)


@pytest.mark.parametrize("task", ["fp", "dn"])
def test_new_application_split_matches_unchanged_shared_helper(tmp_path, task):
    # An odd row count also checks rounding and the second holdout shuffle.
    rows = [{"smiles": f"molecule-{i}", "target": float(i)} for i in range(353)]
    expected = train_val_test_split(Dataset.from_list(rows))
    write_splits(task, rows, [], tmp_path)
    for name, split in expected.items():
        actual = pd.read_csv(tmp_path / f"{name}.csv")
        assert actual.smiles.tolist() == list(split["smiles"])
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["split_counts"] == {k: len(v) for k, v in expected.items()}
    assert protocol["exact_paper_split"] is False
    assert protocol["source_row_order_preserved"] is True
    assert protocol["status"] == "user_supplied_source_shared_split"


# Ordered row IDs from the pre-existing application splitter (3d11da0ca).
# Keep literal fixtures: computing the expectation with the production splitter
# would not catch a change of RNG, rounding, row sorting, or second shuffle.
LEGACY_ORDER = {
    "temperature": {
        "train": [19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11, 2, 16],
        "validation": [7, 8],
        "test": [0, 3, 20],
    },
    "odour": {
        "train": [0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 20, 4, 12],
        "validation": [7, 10],
        "test": [14, 19, 6],
    },
    "solvent": {
        "train": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17, 18, 19],
        "validation": [12, 14, 20],
        "test": [10, 15, 16],
    },
}


@pytest.mark.parametrize("task", ESTABLISHED_TASKS)
def test_existing_application_split_keeps_exact_order(tmp_path, task):
    rows = [{"smiles": f"molecule-{i}", "target": float(i)} for i in range(21)]
    family = (
        "temperature"
        if task in ("bp", "mp")
        else "odour"
        if task == "odour"
        else "solvent"
    )
    write_splits(task, rows, [], tmp_path)
    for name, indices in LEGACY_ORDER[family].items():
        actual = pd.read_csv(tmp_path / f"{name}.csv")
        assert actual.smiles.tolist() == [rows[i]["smiles"] for i in indices]
        assert actual.target.tolist() == [rows[i]["target"] for i in indices]
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["status"] == "public_replacement_not_asserted_exact_paper_split"


def test_shared_training_helper_keeps_exact_order():
    ds = Dataset.from_dict({"row_id": list(range(21))})
    splits = train_val_test_split(ds)
    assert {name: list(split["row_id"]) for name, split in splits.items()} == {
        "train": [14, 7, 6, 9, 3, 0, 17, 5, 11, 19, 2, 4, 18, 1, 13, 8],
        "validation": [10, 16],
        "test": [12, 15, 20],
    }


@pytest.mark.parametrize("task", [*ESTABLISHED_TASKS, "fp", "dn"])
def test_existing_splits_and_provenance_are_not_rewritten(tmp_path, task):
    rows = [{"smiles": f"molecule-{i}", "target": float(i)} for i in range(21)]
    write_splits(task, rows, [{"path": "original-source"}], tmp_path)
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }
    write_splits(task, rows, [{"path": "same-source-moved-on-disk"}], tmp_path)
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    } == before


@pytest.mark.parametrize("change", ["source_order", "split_order", "value", "protocol"])
def test_mismatch_fails_before_touching_any_existing_split(tmp_path, change):
    rows = [{"smiles": f"molecule-{i}", "target": float(i)} for i in range(21)]
    write_splits("pka", rows, [], tmp_path)
    if change == "source_order":
        rows.reverse()
    elif change == "split_order":
        path = tmp_path / "test.csv"
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    elif change == "value":
        rows[0]["target"] += 1
    else:
        path = tmp_path / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["rows"] += 1
        path.write_text(json.dumps(protocol))
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_splits("pka", rows, [], tmp_path)
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    } == before


def test_pka_retains_cached_zenodo_source_and_original_curation(tmp_path):
    source = tmp_path / "pKa_Data_Table-v1.4.0.csv"
    records = [
        {"Canonical_SMILES": "O", "InChIKey": f"key-{i:04d}", "DMSO_pKa": float(i)}
        for i in reversed(range(3001))
    ]
    records.append({"Canonical_SMILES": "O", "InChIKey": "key-0000", "DMSO_pKa": 2.0})
    pd.DataFrame(records).to_csv(source, index=False)
    # A newly supplied alternative must not substitute for the cached release.
    (tmp_path / "pKa_Data_Table.csv").write_text("smiles,DMSO_pKa\nN,99\n")
    before = source.read_bytes()
    rows, provenance = read_cached_source("pka", tmp_path)
    assert [row["DMSO_pKa"] for row in rows] == [1.0, *map(float, range(1, 3001))]
    assert provenance[0]["path"] == str(source)
    assert "zenodo.org/records/16882613/" in provenance[0]["url"]
    assert source.read_bytes() == before


@pytest.mark.parametrize("task", ESTABLISHED_TASKS)
def test_missing_original_sources_do_not_trigger_replacement_downloads(tmp_path, task):
    # The schema is deliberately absent: every cached source must be present,
    # and odour also requires its local label schema.
    error = ValueError if task == "odour" else FileNotFoundError
    with pytest.raises(error):
        read_cached_source(task, tmp_path)


def test_registry_keeps_established_paths_and_split_provenance():
    root = Path(__file__).resolve().parents[1]
    cases = yaml.safe_load((root / "opt/benchmarks/registry.yaml").read_text())["cases"]
    established = [case for case in cases if case.get("dataset") in ESTABLISHED_TASKS]
    assert len(established) == 12
    for case in established:
        task = case["dataset"]
        assert case["path"] == f"../../data/screening-heads/replacement-seed42/{task}"
        assert case["split"] == (
            "public-openpom-seed42" if task == "odour" else "replacement-seed42"
        )


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local dataset preparation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize(
    ("task", "header", "target"),
    [
        ("bp", "iso_smiles,bp", "bp"),
        ("mp", "iso_smiles,mp", "mp"),
        ("fp", "iso_smiles,fp", "fp"),
        ("pka", "smiles,DMSO_pKa", "DMSO_pKa"),
        ("etn", "smiles,E T N (num)", "ETN"),
        ("dn", "smiles,BF3 affinity", "BF3 affinity"),
    ],
)
def test_supplied_headers_and_row_order_are_preserved(tmp_path, task, header, target):
    source = tmp_path / "source.csv"
    # Include duplicate SMILES with distinct measurements; do not average,
    # canonicalize or reorder the author's curation before the shared split.
    content = ("\ufeff" + header + "\r\r\nOCC,2\r\r\nO,1\r\r\nOCC,3\r\r\n").encode(
        "utf-8"
    )
    source.write_bytes(content)
    rows, sources = read_source(task, source)
    assert rows == [
        {"smiles": "OCC", target: 2},
        {"smiles": "O", target: 1},
        {"smiles": "OCC", target: 3},
    ]
    assert sources[0]["path"] == str(source)
    assert source.read_bytes() == content


def test_missing_source_has_no_download_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="no download fallback"):
        read_source("pka", tmp_path / "pKa_Data_Table.csv")


def test_name_only_kt_requires_supplied_smiles(tmp_path):
    source = tmp_path / "Experimental_KT.csv"
    source.write_text("Name,alpha,beta,pi_star\nwater,1.17,0.47,1.09\n")
    with pytest.raises(ValueError, match="Add verified SMILES"):
        read_source("kt", source)


def test_annotated_kt_uses_supplied_smiles_without_name_resolution(tmp_path):
    source = tmp_path / "Experimental_KT.csv"
    source.write_text("Name,smiles,alpha,beta,pi_star\nwater,O,1.17,0.47,1.09\n")
    rows, _ = read_source("kt", source)
    assert rows == [{"smiles": "O", "alpha": 1.17, "beta": 0.47}]


def test_invalid_encoding_does_not_shift_source_split_membership(tmp_path):
    source = tmp_path / "pKa_Data_Table.csv"
    source.write_text("smiles,DMSO_pKa\nND,2\nO,3\n")
    rows, _ = read_source("pka", source)
    assert rows == [{"smiles": "ND", "DMSO_pKa": 2}, {"smiles": "O", "DMSO_pKa": 3}]


@pytest.mark.parametrize("value", ["inf", "wrong-scale", ""])
def test_invalid_measurement_column_fails_explicitly(tmp_path, value):
    source = tmp_path / "dn.csv"
    source.write_text(f"smiles,BF3 affinity\nO,{value}\n")
    with pytest.raises(ValueError):
        read_source("dn", source)
