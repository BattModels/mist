import json
import subprocess
import sys
from pathlib import Path

import pytest

from electrolyte_fm.utils.validate_checksums import snapshot

VALIDATOR = (
    Path(__file__).resolve().parents[1] / "electrolyte_fm/utils/validate_checksums.py"
)


def validate(paths, state=None):
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        input=json.dumps(
            {
                "version": 2,
                "outputs": [{"path": str(path)} for path in paths],
                "validation_state": state,
            }
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_large_tree_has_bounded_state_and_every_file_is_checked(tmp_path):
    for index in range(1000):
        (tmp_path / f"zinc-shard-{index:06d}.txt").write_text("CO\n")
    state = snapshot([tmp_path])
    assert state["files"] == 1000
    assert state["bytes"] == 3000
    assert len(json.dumps(state).encode()) < 1024
    assert validate([tmp_path], state).returncode == 0
    (tmp_path / "zinc-shard-000500.txt").write_text("CN\n")
    assert validate([tmp_path], state).returncode == 2


@pytest.mark.parametrize(
    "mutation", ["content", "remove", "add", "rename", "directory"]
)
def test_recursive_changes_are_invalid(tmp_path, mutation):
    nested = tmp_path / "nested"
    nested.mkdir()
    file = nested / "molecule.txt"
    file.write_text("CO\n")
    state = snapshot([tmp_path])
    if mutation == "content":
        file.write_text("CN\n")
    elif mutation == "remove":
        file.unlink()
    elif mutation == "add":
        (nested / "second.txt").write_text("CC\n")
    elif mutation == "rename":
        file.rename(nested / "renamed.txt")
    else:
        (nested / "empty").mkdir()
    result = validate([tmp_path], state)
    assert result.returncode == 2, result.stderr
    assert not result.stdout


def test_overlapping_outputs_are_hashed_once(tmp_path):
    file = tmp_path / "fragments.sqlite"
    file.write_bytes(b"database")
    state = snapshot([tmp_path])
    assert snapshot([file, tmp_path, file]) == state
    assert state["files"] == 1
    assert state["bytes"] == 8


def test_symlinks_are_recorded_without_following_cycles(tmp_path):
    link = tmp_path / "loop"
    link.symlink_to(tmp_path, target_is_directory=True)
    state = snapshot([tmp_path])
    assert validate([tmp_path], state).returncode == 0
    link.unlink()
    link.symlink_to("missing-target")
    assert validate([tmp_path], state).returncode == 2


def test_cli_returns_state_and_distinguishes_missing_outputs(tmp_path):
    file = tmp_path / "receipt.json"
    file.write_text("{}\n")
    first = validate([file])
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == snapshot([file])
    file.unlink()
    assert validate([file], json.loads(first.stdout)).returncode == 1


def test_output_order_does_not_change_state(tmp_path):
    paths = [tmp_path / name for name in ("a", "b")]
    for path in paths:
        path.write_text(path.name)
    assert snapshot(paths) == snapshot(reversed(paths))
