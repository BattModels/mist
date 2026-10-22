"""QMiST jobs inherit Spack's runtime setup before starting Python."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

QMIST_DIR = Path(__file__).resolve().parents[1] / "opt/qmist"
LAUNCHER = QMIST_DIR / "scripts/with_spack.sh"


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return path


def test_spack_installs_in_the_owning_project_with_native_verified_downloads():
    config = yaml.safe_load((QMIST_DIR / "spack.yaml").read_text())["spack"]["config"]
    assert config["install_tree"]["root"] == "$env/.spack-env/store"
    assert config["source_cache"] == "$env/../../.cache/spack/source"
    assert config["url_fetch_method"] == "curl"
    assert config.get("verify_ssl", True) is True


def test_installer_keeps_uv_downloads_and_python_off_home(tmp_path):
    project = tmp_path / "repo/opt/qmist project"
    project.mkdir(parents=True)
    cache = tmp_path / "repo/.cache"
    installer = project / "install.sh"
    shutil.copyfile(QMIST_DIR / "install.sh", installer)
    executable(
        tmp_path / "bin/uv",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps(dict(os.environ)))\n"
        "sys.exit(23)\n",
    )
    result = subprocess.run(
        ["bash", str(installer)],
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "UV_CACHE_DIR": "/quota-limited/cache",
            "UV_PYTHON_INSTALL_DIR": "/quota-limited/python",
            "UV_PROJECT_ENVIRONMENT": "/wrong/environment",
            "UV_SYSTEM_CERTS": "false",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    # A failed uv sync must stop before any MOPAC download/install.
    assert result.returncode == 23, result.stderr
    env = json.loads(result.stdout.splitlines()[-1])
    assert env["UV_CACHE_DIR"] == str(cache / "uv")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(cache / "python")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(project / ".venv")
    assert env["XDG_CACHE_HOME"] == str(cache)
    assert env["UV_SYSTEM_CERTS"] == "true"
    assert not (project / "vendor").exists()


@pytest.mark.parametrize("existing_library_path", [None, "/existing/lib"])
def test_spack_environment_reaches_python(tmp_path, existing_library_path):
    gaussian = tmp_path / "gaussian runtime"
    executable(gaussian / "g09", "#!/bin/sh\nexit 0\n")
    mopac = tmp_path / "mopac runtime/bin/mopac"
    activation = (
        f'export PATH={shlex.quote(str(gaussian))}:"$PATH"\n'
        f"export LD_LIBRARY_PATH={shlex.quote(str(gaussian))}"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n"
        "export GAUSS_EXEDIR='from spack recipe'\n"
        f"export SPACK_ENV={shlex.quote(str(QMIST_DIR))}\n"
    )
    spack = executable(
        tmp_path / "spack",
        f"#!{sys.executable}\n"
        "import sys\n"
        f"assert sys.argv[1:] == {['env', 'activate', '--sh', '--dir', str(QMIST_DIR)]!r}\n"
        f"print({activation!r})\n",
    )
    inherited_env = {**os.environ, "KEEP_ME": "yes"}
    inherited_env.pop("LD_LIBRARY_PATH", None)
    if existing_library_path is not None:
        inherited_env["LD_LIBRARY_PATH"] = existing_library_path
    result = subprocess.run(
        [
            str(LAUNCHER),
            str(spack),
            str(mopac),
            sys.executable,
            "-c",
            "import json, os, sys; print(json.dumps([dict(os.environ), sys.argv[1:]]))",
            "a molecule with spaces",
        ],
        env=inherited_env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    env, args = json.loads(result.stdout)
    assert env["SPACK_ENV"] == str(QMIST_DIR)
    assert env["GAUSS_EXEDIR"] == "from spack recipe"
    assert env["KEEP_ME"] == "yes"
    assert env["PATH"] == f"{mopac.parent}:{gaussian}:{os.environ['PATH']}"
    library_paths = [f"{mopac.parent}/../lib", str(gaussian)]
    if existing_library_path is not None:
        library_paths.append(existing_library_path)
    assert env["LD_LIBRARY_PATH"] == ":".join(library_paths)
    assert args == ["a molecule with spaces"]


@pytest.mark.parametrize("activation_exit", [0, 23])
def test_python_does_not_start_without_spack_gaussian(tmp_path, activation_exit):
    spack = executable(tmp_path / "spack", f"#!/bin/sh\nexit {activation_exit}\n")
    result = subprocess.run(
        [
            str(LAUNCHER),
            str(spack),
            str(tmp_path / "mopac/bin/mopac"),
            sys.executable,
            "-c",
            "print('Python must not start')",
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (activation_exit or 1)
    assert not result.stdout
    if not activation_exit:
        assert "Spack did not make g09 available" in result.stderr
