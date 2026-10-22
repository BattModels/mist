"""Local checks for accepted HF data staging and unchanged fold evaluation."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, Features, Value

from opt.mixtures import materialize_dataset as materialize


def saved_split(path, identifiers):
    columns = {
        "doi": [str(identifier) for identifier in identifiers],
        "smi1": ["CC"] * len(identifiers),
        "smi2": ["O"] * len(identifiers),
        "x1": [0.5] * len(identifiers),
        "x2": [0.5] * len(identifiers),
        "temperature [kelvin]": [298.15] * len(identifiers),
        "pressure [megapascal]": [0.1] * len(identifiers),
    }
    for name in [
        "density [gram / centimeter ** 3]",
        "molar volume [centimeter ** 3 / mole]",
        "molar enthalpy [joule / mole]",
    ]:
        columns[name] = [float(identifier) for identifier in identifiers]
        columns["excess " + name] = [None] * len(identifiers)
    features = Features(
        {
            name: Value("string" if name in {"doi", "smi1", "smi2"} else "float64")
            for name in columns
        }
    )
    Dataset.from_dict(columns, features=features).save_to_disk(path)


def test_table_export_preserves_all_random_splits_duplicates_and_nulls(tmp_path):
    for split, ids in [("train", [1, 1]), ("validation", [2]), ("test", [3])]:
        saved_split(tmp_path / "source/random" / split, ids)
    output = tmp_path / "database.csv"
    assert materialize.export_table(tmp_path / "source", output) == 4
    table = pd.read_csv(output)
    assert table.doi.tolist() == [1, 1, 2, 3]
    assert table["density [gram / centimeter ** 3]"].tolist() == [1, 1, 2, 3]
    assert table["excess density [gram / centimeter ** 3]"].isna().all()
    assert set(table) >= {"x1", "x2", "pressure [megapascal]"}


def test_native_evaluation_accepts_published_two_split_fold(tmp_path):
    for split, ids in [("train", [1, 2]), ("validation", [3])]:
        saved_split(tmp_path / split, ids)
    source = Path(__file__).resolve().parents[1] / "python/excess.py"
    spec = importlib.util.spec_from_file_location("mixture_excess_dataset_test", source)
    excess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(excess)
    data = excess.ComponentDataModuleFast.__new__(
        excess.ComponentDataModuleFast
    )
    data.path = tmp_path
    splits = data._get_dataset()
    assert set(splits) == {"train", "validation"}
    assert [row["doi"] for row in splits["validation"]] == ["3"]
    (tmp_path / "test").mkdir()
    with pytest.raises(FileNotFoundError, match="test"):
        data._get_dataset()


def test_staging_uses_dataset_repository_and_exports_directory_link(
    tmp_path, monkeypatch
):
    import huggingface_hub

    snapshot = tmp_path / materialize.REVISION
    for split in ("train", "validation", "test"):
        directory = snapshot / "random" / split
        directory.mkdir(parents=True)
        (directory / "data.arrow").write_bytes(b"not loaded in staging")
    observed = {}

    def download(repo_id, **kwargs):
        observed.update(repo_id=repo_id, **kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(materialize, "CONFIGURATIONS", ["random"])
    output = tmp_path / "staged-data"
    materialize.stage(output, offline=True)
    assert observed["repo_id"] == "mist-models/excess-properties"
    assert observed["repo_type"] == "dataset"
    assert observed["revision"] == materialize.REVISION
    assert observed["local_files_only"] is True
    assert output.is_symlink() and output.resolve() == snapshot
