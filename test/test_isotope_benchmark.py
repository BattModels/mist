"""The accepted isotope source uses existing parsing, target values and splitting."""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from opt.benchmarks.scripts.materialize_isotopes import materialize
from opt.isotopes.prepare import read_cohort, read_protocol

ROOT = Path(__file__).resolve().parents[1]


def wolfram_source(path):
    names = [f"carbon-{index}" for index in range(10, 30)] + ["carbon-14m", "carbon-14m1", "carbon-14m2", "carbon-14m3", "carbon-14"]
    rows = [["Association", ["Rule", "'Isotope'", f"'{name}'"],
             ["Rule", "'Half Life'", 1.25 + index], ["Rule", "'log(Time Scale)'", index % 5 - 2]]
            for index, name in enumerate(names)]
    path.write_text(json.dumps(["Dataset", ["List", *rows]]))
    return names


def test_materializer_keeps_original_suffix_filter_targets_order_and_duplicate_rows(tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "dataset"
    names = wolfram_source(source)
    materialize(source, output, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    cohort = pd.read_csv(output / "cohort.csv")
    assert cohort.name.tolist() == names[:20] + names[-2:]
    assert cohort.half_life.tolist() == [1.25 + index for index in range(20)] + [24.25, 25.25]
    assert cohort.log_time.tolist() == [index % 5 - 2 for index in range(20)] + [1, 2]
    assert len(read_cohort(output / "cohort.csv")) == 22
    splits = {name: pd.read_csv(output / f"{name}.csv") for name in ("train", "validation", "test")}
    assert {row for frame in splits.values() for row in frame.row_id} == set(cohort.row_id)
    assert sum(len(frame) for frame in splits.values()) == len(cohort)
    for name, frame in splits.items():
        assert set(frame.row_id) == set(cohort.loc[cohort.split == name, "row_id"])
    stat = (output / "train.csv").stat()
    materialize(source, output)
    assert (output / "train.csv").stat().st_mtime_ns == stat.st_mtime_ns
    (output / "test.csv").write_text("different established split")
    with pytest.raises(FileExistsError, match="Existing isotope artifact differs"):
        materialize(source, output)


def test_materializer_rejects_changed_pinned_source_before_writing(tmp_path):
    source = tmp_path / "source.json"
    wolfram_source(source)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        materialize(source, tmp_path / "output", sha256="0" * 64)
    assert not (tmp_path / "output").exists()


def test_materializer_downloads_missing_source_and_reuses_it(tmp_path):
    published = tmp_path / "published.json"
    wolfram_source(published)
    payload = published.read_bytes()
    source, output = tmp_path / "cache/source.json", tmp_path / "dataset"
    options = {"source_url": published.as_uri(), "sha256": hashlib.sha256(payload).hexdigest()}
    materialize(source, output, **options)
    assert source.read_bytes() == payload
    assert len(pd.read_csv(output / "cohort.csv")) == 22
    paths = [source, *output.iterdir()]
    mtimes = {path: path.stat().st_mtime_ns for path in paths}
    published.unlink()
    materialize(source, output, **options)
    assert {path: path.stat().st_mtime_ns for path in paths} == mtimes


def test_materializer_rejects_incorrect_download_without_publishing_it(tmp_path):
    published = tmp_path / "published.json"
    wolfram_source(published)
    source, output = tmp_path / "cache/source.json", tmp_path / "dataset"
    with pytest.raises(subprocess.CalledProcessError):
        materialize(source, output, source_url=published.as_uri(), sha256="0" * 64)
    assert not source.exists()
    assert not output.exists()


def test_checked_in_isotope_projection_preserves_original_plot_defaults():
    protocol = read_protocol(ROOT / "opt/isotopes/protocol.json")
    assert protocol == dict(encoding="smiles", pooling="cls", projection_columns="legacy_numeric",
                            n_neighbors=10, min_dist=2, metric="manhattan", seed=42,
                            tol_offset=0.8, tol_scale=0.02, umap_kwargs={})
