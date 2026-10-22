"""Workflow caches belong to the checkout, not to its projects or launch cwd."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from electrolyte_fm.utils import stage_hf, sync_env

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE_PATHS = {
    "HF_HOME": CACHE / "huggingface",
    "XDG_CACHE_HOME": CACHE,
    "UV_CACHE_DIR": CACHE / "uv",
    "UV_PYTHON_INSTALL_DIR": CACHE / "python",
    "SPACK_USER_CACHE_PATH": CACHE / "spack",
}


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia CLI required")
def test_julia_depot_keeps_home_out_of_cache_search(tmp_path):
    depot = tmp_path / "repo/.cache/julia"
    result = subprocess.run(
        [
            "julia",
            "--startup-file=no",
            "--history-file=no",
            "-e",
            "println.(DEPOT_PATH)",
        ],
        cwd=tmp_path,
        env={**os.environ, "JULIA_DEPOT_PATH": str(depot) + os.pathsep},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    paths = [Path(line) for line in result.stdout.splitlines()]
    assert paths[0] == depot
    assert Path.home() / ".julia" not in paths
    assert len(paths) > 1  # Julia's installed system depots remain available.


@pytest.mark.skipif(
    shutil.which("julia") is None or shutil.which("pomset") is None,
    reason="Julia and Pomset CLIs required",
)
@pytest.mark.parametrize("working_directory", [".", "opt/screening"])
def test_pomset_renders_typed_julia_depot_with_system_depots(
    tmp_path, working_directory
):
    shutil.copyfile(ROOT / "config.pomset", tmp_path / "config.pomset")
    (tmp_path / ".pomset").mkdir()
    working = tmp_path / working_directory
    working.mkdir(parents=True, exist_ok=True)
    (tmp_path / "workflow.pomset").write_text(
        """load("//config.pomset", "ENV")
task(
    name="julia-depot",
    cmd=[
        "julia", "--startup-file=no", "--history-file=no", "-e",
        'mkpath(dirname(ARGS[1])); write(ARGS[1], ENV["JULIA_DEPOT_PATH"]); write(ARGS[2], join(DEPOT_PATH, "\\n"))',
        Path("out/environment.txt"), Path("out/depots.txt"),
    ],
    env=ENV,
    executor=shell(),
    working_directory=Path("."),
    output_dir=Path("out"),
    outputs={"environment": Path("environment.txt"), "depots": Path("depots.txt")},
)
""".replace(
            'working_directory=Path(".")',
            f'working_directory=Path("{working_directory}")',
        )
    )
    result = subprocess.run(
        ["pomset", "submit", "--backend", "local", "--no-progress"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    depot = tmp_path / ".cache/julia"
    assert (tmp_path / "out/environment.txt").read_text() == (
        os.path.relpath(depot, working) + os.pathsep
    )
    paths = [
        Path(line) for line in (tmp_path / "out/depots.txt").read_text().splitlines()
    ]
    assert (working / paths[0]).resolve() == depot
    assert Path.home() / ".julia" not in paths
    assert len(paths) > 1


@pytest.mark.parametrize("cwd", [".", "opt/benchmarks", "opt/FeatureMiner"])
def test_sync_shares_root_cache_but_keeps_project_venv(tmp_path, monkeypatch, cwd):
    project = tmp_path / "project"
    project.mkdir()
    launch = ROOT / cwd
    if not launch.is_dir():
        launch = tmp_path
    monkeypatch.chdir(launch)
    monkeypatch.setenv("POMSET_OUTPUT_DIR", str(tmp_path / "receipt"))
    for name in CACHE_PATHS:
        monkeypatch.setenv(name, "/wrong/parent/cache")
    monkeypatch.setenv("UV_SYSTEM_CERTS", "false")
    monkeypatch.setenv("JULIA_DEPOT_PATH", "/wrong/parent/julia")
    monkeypatch.setattr(sys, "argv", ["sync_env.py", "--project", str(project)])
    monkeypatch.setattr(sync_env.shutil, "which", lambda _: "/usr/bin/uv")
    calls = []

    def run(command, *, env, check):
        calls.append(command)
        assert check
        for name, path in CACHE_PATHS.items():
            if name != "SPACK_USER_CACHE_PATH":
                assert env[name] == str(path)
        assert env["UV_PROJECT_ENVIRONMENT"] == str(project / ".venv")
        assert env["UV_SYSTEM_CERTS"] == "true"
        assert env["JULIA_DEPOT_PATH"] == str(CACHE / "julia") + os.pathsep

    monkeypatch.setattr(sync_env.subprocess, "run", run)
    sync_env.main()
    assert calls == [["/usr/bin/uv", "sync", "--frozen", "--project", str(project)]]
    receipt = json.loads((tmp_path / "receipt/receipt.json").read_text())
    assert receipt["cache"] == str(CACHE / "uv")


def test_stage_snapshot_uses_root_cache_and_links_actual_location(
    tmp_path, monkeypatch
):
    assert stage_hf.CACHE == CACHE
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", "/wrong/parent/cache")
    monkeypatch.setenv("HF_HUB_CACHE", "/another/wrong/cache")
    snapshot = tmp_path / "models--example--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    output = tmp_path / "artifacts/model"
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        assert kwargs["cache_dir"] == str(CACHE / "huggingface/hub")
        assert kwargs["local_files_only"] is True
        return str(snapshot)

    monkeypatch.setattr(stage_hf, "snapshot_download", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_hf.py",
            "example/model",
            "--revision",
            "pinned-revision",
            "--offline",
            "--output",
            str(output),
        ],
    )
    stage_hf.main()
    assert len(calls) == 1
    assert output.is_symlink()
    assert output.resolve() == snapshot
    assert (output / "config.json").read_text() == "{}"


def test_qmist_bootstrap_uses_repo_cache_from_an_unrelated_cwd(tmp_path):
    project = tmp_path / "repo/opt/qmist"
    project.mkdir(parents=True)
    shutil.copyfile(ROOT / "opt/qmist/install.sh", project / "install.sh")
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps(dict(os.environ)))\n"
        "sys.exit(23)\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(project / "install.sh")],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 23, result.stderr
    env = json.loads(result.stdout.splitlines()[-1])
    cache = tmp_path / "repo/.cache"
    assert env["UV_CACHE_DIR"] == str(cache / "uv")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(cache / "python")
    assert env["XDG_CACHE_HOME"] == str(cache)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(project / ".venv")
    assert env["UV_SYSTEM_CERTS"] == "true"


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
@pytest.mark.parametrize(
    "workflow",
    ["workflow.pomset"]
    + [
        f"opt/{name}/workflow.pomset"
        for name in (
            "benchmarks",
            "sterochemistry",
            "screening",
            "synth_access",
            "FeatureMiner",
            "design",
        )
        if (ROOT / f"opt/{name}/workflow.pomset").is_file()
    ],
)
def test_pomset_resolves_all_declared_caches_to_repo_root(inspect_workflow, workflow):
    # Only copied definitions and fresh temporary state are inspected.
    tasks = inspect_workflow(workflow)
    checked = 0
    for task in tasks:
        if task["id"] == "benchmark-pubchemqc-materialize":
            assert "HF_HOME" in task["environment"]
            assert "XDG_CACHE_HOME" in task["environment"]
        depot = task["environment"].get("JULIA_DEPOT_PATH")
        if depot is not None:
            value = depot["Path"]
            assert value["rendering"] == "resolved"
            rendered = str(ROOT / value["origin"]["value"] / value["path"]["value"])
            writable, system = rendered.split(":")
            assert Path(writable) == CACHE / "julia"
            assert system == ""  # Include bundled depots, never the home depot.
        for name, expected in CACHE_PATHS.items():
            expression = task["environment"].get(name)
            if expression is None:
                continue
            value = expression["Path"]
            assert value["rendering"] == "resolved"
            actual = (
                ROOT / value["origin"]["value"] / value["path"]["value"]
            ).resolve()
            assert actual == expected, (task["id"], name, actual)
            checked += 1
    assert checked > 0
