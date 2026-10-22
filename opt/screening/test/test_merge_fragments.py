import os
import select
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fragment_zinc as merge

sql_merge_pair = merge.merge_pair


def merge_with_straggler(lhs, rhs, directory):
    """Hold the first pair until two other completed pairs merge again."""
    followup = lhs.parent / "followup-completed"
    (lhs.parent / f"worker-{os.getpid()}").touch()
    if lhs.name == "0.sqlite":
        deadline = time.monotonic() + 20
        while not followup.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("completed pairs waited for the straggler")
            time.sleep(0.01)
        if (lhs.parent / "interrupt").exists():
            raise RuntimeError("interrupted straggler")
    result = sql_merge_pair(lhs, rhs, directory)
    if {lhs.name, rhs.name} == {"2.sqlite", "4.sqlite"}:
        followup.touch()
    return result


def database(path, fragments):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE fragments (id INTEGER PRIMARY KEY, fragment TEXT UNIQUE)"
        )
        connection.executemany(
            "INSERT INTO fragments (fragment) VALUES (?)", ((s,) for s in fragments)
        )
    return path


def contents(path):
    with closing(sqlite3.connect(path)) as connection:
        return {row[0] for row in connection.execute("SELECT fragment FROM fragments")}


@pytest.fixture
def primary(tmp_path):
    directory = tmp_path / "primary"
    directory.mkdir()
    return directory


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_merge_of_odd_number_of_databases(tmp_path, workers):
    for index, fragments in enumerate(
        [("CC", "CO"), ("CO", "CN"), ("CF",), ("CC", "CS"), ("CP",)]
    ):
        database(tmp_path / f"{index}.sqlite", fragments)
    output = tmp_path / "result.sqlite"
    output.write_bytes(b"previous output")
    assert merge.merge_fragments(tmp_path, output, workers) == output
    assert contents(output) == {"CC", "CO", "CN", "CF", "CS", "CP"}
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("interrupt", [False, True])
def test_completed_pairs_merge_before_straggler_finishes(
    tmp_path, monkeypatch, interrupt
):
    fragments = {f"fragment-{index}" for index in range(6)}
    for index in range(6):
        database(tmp_path / f"{index}.sqlite", [f"fragment-{index}", "shared"])
    output = tmp_path / "result.sqlite"
    if interrupt:
        (tmp_path / "interrupt").touch()

    with monkeypatch.context() as patch:
        patch.setattr(merge, "merge_pair", merge_with_straggler)
        if interrupt:
            with pytest.raises(RuntimeError, match="interrupted straggler"):
                merge.merge_fragments(tmp_path, output, workers=3)
        else:
            merge.merge_fragments(tmp_path, output, workers=3)

    assert (tmp_path / "followup-completed").exists()
    assert len(list(tmp_path.glob("worker-*"))) == 3
    assert not (tmp_path / f"worker-{os.getpid()}").exists()
    if interrupt:
        assert not output.exists()
        # The completed follow-up is a checkpoint even though the first pair
        # never finished. Resume from it and the two untouched inputs.
        assert len(list(tmp_path.glob("*.sqlite"))) == 3
        assert contents(tmp_path / "0.sqlite") == {"fragment-0", "shared"}
        assert contents(tmp_path / "1.sqlite") == {"fragment-1", "shared"}
        merge.merge_fragments(tmp_path, output, workers=3)
    assert contents(output) == fragments | {"shared"}
    assert list(tmp_path.glob("*.sqlite")) == [output]


def test_pair_is_one_sql_insert_on_local_databases(tmp_path, monkeypatch, primary):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CC", "CO"])
    original = sqlite3.connect
    statements = []

    class SQLOnly(sqlite3.Connection):
        def execute(self, statement, parameters=()):
            statements.append(statement)
            assert not statement.startswith("SELECT")
            if statement.startswith("INSERT"):
                assert "ORDER BY fragment" in statement
                paths = [
                    Path(row[2]) for row in super().execute("PRAGMA database_list")
                ]
                assert {path.parent for path in paths} == {primary}
            return super().execute(statement, parameters)

        def executemany(self, *args, **kwargs):
            pytest.fail("Production merge must not iterate through fragment rows")

    with monkeypatch.context() as patch:
        patch.setattr(
            merge.sqlite3,
            "connect",
            lambda *a, **kw: original(*a, **kw, factory=SQLOnly),
        )
        merge.merge_pair(lhs, rhs, primary)
    assert len([s for s in statements if s.startswith("INSERT")]) == 1
    assert contents(lhs) == {"CC", "CO"}
    assert not rhs.exists()


def test_failed_pair_rolls_back_and_retains_rhs(tmp_path, monkeypatch, primary):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CO"])
    original = sqlite3.connect

    class Interrupted(sqlite3.Connection):
        def execute(self, statement, parameters=()):
            result = super().execute(statement, parameters)
            if statement.startswith("INSERT"):
                raise RuntimeError("interrupted before commit")
            return result

    with monkeypatch.context() as patch:
        patch.setattr(
            merge.sqlite3,
            "connect",
            lambda *a, **kw: original(*a, **kw, factory=Interrupted),
        )
        with pytest.raises(RuntimeError):
            merge.merge_pair(lhs, rhs, primary)
    assert contents(lhs) == {"CC"}
    assert contents(rhs) == {"CO"}


def test_retry_after_checkpoint_before_rhs_deletion(tmp_path, monkeypatch, primary):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CO"])

    unlink = Path.unlink

    def interrupted_deletion(path, **kwargs):
        if path == rhs:
            raise RuntimeError("interrupted after checkpoint")
        return unlink(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted_deletion)
        with pytest.raises(RuntimeError):
            merge.merge_pair(lhs, rhs, primary)
    assert contents(lhs) == {"CC", "CO"}
    assert rhs.exists()
    output = tmp_path / "result.sqlite"
    merge.merge_fragments(tmp_path, output, workers=2)
    assert contents(output) == {"CC", "CO"}


def test_retry_rebuilds_primary_on_fresh_node(tmp_path, primary):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", [f"fragment-{i}" for i in range(10000)])
    code = """
import os, signal, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('PRAGMA cache_size=1')
db.execute('ATTACH DATABASE ? AS rhs', (sys.argv[2],))
db.execute('INSERT OR IGNORE INTO fragments (fragment) SELECT fragment FROM rhs.fragments')
os.write(1, b'uncommitted\\n')
signal.pause()
"""
    target_lhs = merge.stage_database(lhs, primary)
    target_rhs = merge.stage_database(rhs, primary)
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(target_lhs), str(target_rhs)],
        stdout=subprocess.PIPE,
    )
    try:
        assert select.select([process.stdout], [], [], 20)[0]
        assert process.stdout.readline() == b"uncommitted\n"
    finally:
        process.kill()
        process.communicate(timeout=20)
    # The next allocation has a fresh /tmp and starts from the checkpoints.
    shutil.rmtree(primary)
    output = tmp_path / "result.sqlite"
    merge.merge_fragments(tmp_path, output, workers=2)
    assert contents(output) == {"CC"} | {f"fragment-{i}" for i in range(10000)}


def test_primary_is_copied_once_and_checkpointed_after_each_merge(
    tmp_path, primary, monkeypatch
):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CO"])
    third = database(tmp_path / "c.sqlite", ["CN"])
    copyfile = shutil.copyfile
    copies = []

    def record_copy(source, destination):
        copies.append((source, destination))
        return copyfile(source, destination)

    monkeypatch.setattr(merge.shutil, "copyfile", record_copy)
    merge.merge_pair(lhs, rhs, primary)
    assert contents(lhs) == {"CC", "CO"}
    primary_lhs = primary / lhs.name
    # Subsequent work must use the primary, even if it has changed since the
    # previous checkpoint was published.
    with closing(sqlite3.connect(primary_lhs)) as connection, connection:
        connection.execute("INSERT INTO fragments (fragment) VALUES ('CF')")
    assert contents(lhs) == {"CC", "CO"}
    merge.merge_pair(lhs, third, primary)
    assert contents(lhs) == contents(primary_lhs) == {"CC", "CO", "CN", "CF"}
    assert [source for source, _ in copies if source.parent == tmp_path] == [
        lhs,
        rhs,
        third,
    ]
    assert sum(source == primary_lhs for source, _ in copies) == 2
    assert list(primary.glob("*.sqlite")) == [primary_lhs]
    assert list(tmp_path.glob("*.sqlite")) == [lhs]


@pytest.mark.parametrize("failure", ["stage", "checkpoint"])
def test_failed_copy_preserves_checkpoints_for_a_fresh_node(
    tmp_path, primary, monkeypatch, failure
):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CO"])
    copyfile = shutil.copyfile

    def fail_copy(source, destination):
        if destination.parent == (primary if failure == "stage" else tmp_path):
            destination.write_bytes(b"interrupted copy")
            raise OSError("copy failed")
        return copyfile(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(merge.shutil, "copyfile", fail_copy)
        with pytest.raises(OSError, match="copy failed"):
            merge.merge_pair(lhs, rhs, primary)
    assert contents(lhs) == {"CC"}
    assert contents(rhs) == {"CO"}
    assert not list(tmp_path.rglob("*.sqlite.partial"))
    shutil.rmtree(primary)
    output = tmp_path / "result.sqlite"
    merge.merge_fragments(tmp_path, output, workers=2)
    assert contents(output) == {"CC", "CO"}


def test_restart_after_process_is_killed_during_checkpoint_copy(tmp_path, primary):
    lhs = database(tmp_path / "a.sqlite", ["CC"])
    rhs = database(tmp_path / "b.sqlite", ["CO"])
    code = """
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import fragment_zinc
copy_database = fragment_zinc.copy_database
def interrupted_copy(source, destination):
    if destination == Path(sys.argv[2]):
        destination.with_suffix('.sqlite.partial').write_bytes(b'incomplete checkpoint')
        os.write(1, b'copying\\n')
        signal.pause()
    return copy_database(source, destination)
fragment_zinc.copy_database = interrupted_copy
fragment_zinc.merge_pair(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(Path(merge.__file__).parent),
            str(lhs),
            str(rhs),
            str(primary),
        ],
        stdout=subprocess.PIPE,
    )
    try:
        assert select.select([process.stdout], [], [], 20)[0]
        assert process.stdout.readline() == b"copying\n"
    finally:
        process.kill()
        process.communicate(timeout=20)
    assert contents(lhs) == {"CC"}
    assert contents(rhs) == {"CO"}
    assert lhs.with_suffix(".sqlite.partial").exists()
    shutil.rmtree(primary)
    output = tmp_path / "result.sqlite"
    merge.merge_fragments(tmp_path, output, workers=2)
    assert contents(output) == {"CC", "CO"}
    assert list(tmp_path.iterdir()) == [output]
