import csv
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import Future
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fragment_zinc
from src import fragment


@pytest.mark.parametrize(
    ("suffix", "text"),
    [
        (".txt", "CC ZINC1\nCO ZINC2\n"),
        (".smi", "CC\tZINC1\nCO\tZINC2\n"),
        (".csv", "smi,name\nCC,ZINC1\nCO,ZINC2\n"),
    ],
)
def test_fragment_input_is_tab_separated(tmp_path, monkeypatch, suffix, text):
    source = tmp_path / f"input{suffix}"
    source.write_text(text)
    fragmenter = tmp_path / "fasmifra_fragment.py"

    def run(command, *, check):
        assert check is True
        assert command[:2] == [sys.executable, fragmenter]
        assert "-w" not in command
        with Path(command[command.index("-i") + 1]).open() as handle:
            assert list(csv.reader(handle, delimiter="\t")) == [
                ["CC", "ZINC1"],
                ["CO", "ZINC2"],
            ]

    monkeypatch.setattr(fragment.subprocess, "run", run)
    fragment.fragment_smi(source, str(tmp_path / "output.frag"), fragmenter=fragmenter)


@pytest.fixture
def inputs(tmp_path):
    zinc = tmp_path / "zinc"
    for split, contents in (
        ("train", "CC ZINC1\nCO ZINC2\n"),
        ("val", "CO\tZINC2\nCN\tZINC3\n"),
        ("test", "CF ZINC4\nCC ZINC1\n"),
    ):
        (zinc / split).mkdir(parents=True)
        (zinc / split / "part.txt").write_text(contents)
    fragmenter = tmp_path / "fasmifra_fragment.py"
    fragmenter.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "source = pathlib.Path(sys.argv[sys.argv.index('-i') + 1])\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "with source.open() as reader, output.open('w') as writer:\n"
        "    for line in reader:\n"
        "        smiles, name = line.strip().split('\\t')\n"
        "        writer.write(smiles + '\\n')\n"
    )
    # The current interpreter must launch the script without execute permission.
    fragmenter.chmod(0o644)
    return zinc, fragmenter, tmp_path / "work", tmp_path / "output.sqlite"


def contents(path):
    with closing(sqlite3.connect(path)) as database:
        return {row[0] for row in database.execute("SELECT fragment FROM fragments")}


def test_fragment_import_and_merge(inputs):
    _, _, work, output = inputs
    fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert contents(output) == {"CC", "CO", "CN", "CF"}
    assert len(list(work.rglob("*.frag"))) == 3
    assert not list(work.rglob("*.sqlite"))
    assert not (work / "imports").exists()
    assert not list(work.rglob("*.gz"))
    with closing(sqlite3.connect(output)) as database:
        assert database.execute(
            "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM fragments"
        ).fetchone() == (4, 1, 4)
        assert database.execute("PRAGMA index_list(fragments)").fetchall() == []


def test_shards_run_concurrently_in_separate_processes(inputs):
    _, fragmenter, work, _ = inputs
    # Fragmenter subprocesses must have distinct worker parents. The rendezvous
    # requires two shards to be in flight at once, without timing their speed.
    command = fragmenter.read_text().replace(
        "with source.open() as reader, output.open('w') as writer:",
        "import os, time\n"
        "(output.parent / 'worker.pid').write_text(str(os.getppid()))\n"
        "deadline = time.monotonic() + 20\n"
        "while True:\n"
        "    parents = {p.read_text() for p in output.parents[1].glob('*/worker.pid')}\n"
        "    parents.discard('')\n"
        "    if len(parents) >= 2:\n"
        "        break\n"
        "    if time.monotonic() > deadline:\n"
        "        raise RuntimeError('shards are not running in separate processes')\n"
        "    time.sleep(0.01)\n"
        "with source.open() as reader, output.open('w') as writer:",
    )
    fragmenter.write_text(command)
    fragment_zinc.fragment_zinc(*inputs, workers=2)
    parents = {int(p.read_text()) for p in work.glob("fragments/*/worker.pid")}
    assert len(parents) == 2
    assert os.getpid() not in parents
    assert contents(inputs[-1]) == {"CC", "CO", "CN", "CF"}


def test_resume_preserves_completed_shard_imports(inputs):
    zinc, fragmenter, work, output = inputs
    imports = work / "imports"
    imports.mkdir(parents=True)
    completed = work / "fragments/000000"
    source = sorted(zinc.rglob("*.txt"))[0]
    fragment_zinc.fragment_shard(source, completed, imports, fragmenter)
    fragments = completed / "fragments.frag"
    timestamp = fragments.stat().st_mtime_ns
    source.write_text("changed ZINC_CHANGED\n")

    fragment_zinc.fragment_zinc(*inputs, workers=2)

    assert fragments.stat().st_mtime_ns == timestamp
    assert contents(output) == {"CC", "CO", "CN", "CF"}


def test_fragmenter_failure_stops_before_merge(inputs):
    _, fragmenter, work, output = inputs
    fragmenter.write_text(f"#!{sys.executable}\nraise SystemExit(7)\n")
    with pytest.raises(subprocess.CalledProcessError):
        fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert not output.exists()
    assert not (work / "merge").exists()


def test_fragmenter_failure_cancels_queued_shards(inputs, monkeypatch):
    submitted = []

    class Executor:
        def __init__(self, *, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            future = Future()
            if not submitted:
                future.set_exception(RuntimeError("fragmenter failed"))
            submitted.append(future)
            return future

    monkeypatch.setattr(fragment_zinc, "ProcessPoolExecutor", Executor)
    with pytest.raises(RuntimeError, match="fragmenter failed"):
        fragment_zinc.fragment_zinc(*inputs, workers=1)
    assert len(submitted) == 3
    assert all(future.cancelled() for future in submitted[1:])
    assert not inputs[-1].exists()
    assert not (inputs[-2] / "merge").exists()


def test_resume_merge_does_not_recreate_completed_inputs(inputs, monkeypatch):
    _, _, work, output = inputs

    def interrupted(directory, output, workers):
        lhs, rhs, _ = sorted(directory.glob("*.sqlite"))
        with TemporaryDirectory(dir="/tmp") as primary:
            fragment_zinc.merge_pair(lhs, rhs, Path(primary))
        raise RuntimeError("interrupted after completed pair")

    with monkeypatch.context() as patch:
        patch.setattr(fragment_zinc, "merge_fragments", interrupted)
        with pytest.raises(RuntimeError, match="completed pair"):
            fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert len(list((work / "merge").glob("*.sqlite"))) == 2

    def unexpected(*args, **kwargs):
        pytest.fail("A merge retry must not rerun fragmentation or imports")

    monkeypatch.setattr(fragment_zinc, "fragment_shard", unexpected)
    fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert contents(output) == {"CC", "CO", "CN", "CF"}


def test_completed_output_is_not_rebuilt(inputs, monkeypatch):
    fragment_zinc.fragment_zinc(*inputs, workers=2)

    def unexpected(*args, **kwargs):
        pytest.fail("Completed output must not be rebuilt")

    monkeypatch.setattr(fragment_zinc, "merge_fragments", unexpected)
    monkeypatch.setattr(fragment_zinc, "fragment_shard", unexpected)
    monkeypatch.setattr(fragment_zinc, "finalize_database", unexpected)
    assert fragment_zinc.fragment_zinc(*inputs, workers=2) == inputs[-1]


@pytest.mark.parametrize("source_kind", ["checkpoint", "legacy_output"])
def test_interrupted_finalization_preserves_source_and_resumes(
    inputs, monkeypatch, source_kind
):
    _, _, work, output = inputs
    source = output
    if source_kind == "checkpoint":
        source = work / "merge/000000.sqlite"
        source.parent.mkdir(parents=True)
    with closing(sqlite3.connect(source)) as database, database:
        database.execute(
            "CREATE TABLE fragments (fragment TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        database.executemany(
            "INSERT INTO fragments VALUES (?)", [("CC",), ("CO",), ("CN",)]
        )
    original = source.read_bytes()
    pending = output.with_suffix(".sqlite.partial")
    replace = Path.replace

    def interrupted_publication(path, destination):
        if path == pending:
            raise OSError("interrupted final publication")
        return replace(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupted_publication)
        with pytest.raises(OSError, match="interrupted final publication"):
            fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert source.read_bytes() == original
    if source_kind == "checkpoint":
        assert not output.exists()

    # A requeue can leave an incomplete destination. Rebuild it from the source.
    pending.write_bytes(b"incomplete final database")

    def unexpected(*args, **kwargs):
        pytest.fail("Finalization must not repeat fragmentation or shard imports")

    monkeypatch.setattr(fragment_zinc, "fragment_shard", unexpected)
    fragment_zinc.fragment_zinc(*inputs, workers=2)
    assert contents(output) == {"CC", "CO", "CN"}
    with closing(sqlite3.connect(output)) as database:
        assert database.execute(
            "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM fragments"
        ).fetchone() == (3, 1, 3)
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert not pending.exists()
    assert not list(work.rglob("*.sqlite"))
