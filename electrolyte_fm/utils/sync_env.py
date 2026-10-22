#!/usr/bin/env python3
"""Synchronize a shared uv environment on a scheduler head node."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def bindgen_environment(process_env):
    """Let Rust bindgen find the C compiler's standard headers on HPC hosts."""
    compiler = shlex.split(process_env.get("CC", "cc"))
    result = subprocess.run(  # noqa: UP022 - bootstrap runs on Python 3.6.
        [*compiler, "-print-file-name=include"],
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # noqa: UP021 - Python 3.6 has no text argument.
        check=True,
    )
    include = Path(result.stdout.strip())
    if not include.is_absolute() or not (include / "stddef.h").is_file():
        raise FileNotFoundError(
            f"C compiler did not report its standard headers: {include}"
        )
    flags = shlex.split(process_env.get("BINDGEN_EXTRA_CLANG_ARGS", ""))
    flags.extend(["-isystem", str(include)])
    # Some GCC installations keep limits.h in a separate fixed-header tree.
    fixed = include.parent / "include-fixed"
    if fixed.is_dir():
        flags.extend(["-isystem", str(fixed)])
    return {
        **process_env,
        "BINDGEN_EXTRA_CLANG_ARGS": " ".join(
            shlex.quote(flag) for flag in flags
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bindgen-headers", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    environment = project / ".venv"
    # Share downloads across projects, independent of the invocation directory.
    # Only the virtual environment itself belongs to the individual project.
    cache = Path(__file__).resolve().parents[2] / ".cache"
    process_env = dict(
        os.environ,
        UV_PROJECT_ENVIRONMENT=str(environment),
        UV_CACHE_DIR=str(cache / "uv"),
        # Managed interpreters use a separate uv store, not UV_CACHE_DIR.
        # Keep both downloads and installed runtimes off quota-limited HOME.
        UV_PYTHON_INSTALL_DIR=str(cache / "python"),
        XDG_CACHE_HOME=str(cache),
        HF_HOME=str(cache / "huggingface"),
        JULIA_DEPOT_PATH=str(cache / "julia") + os.pathsep,
        # uv is a native executable; the bootstrap itself may be Python 3.6.
        UV_SYSTEM_CERTS="true",
    )
    if args.bindgen_headers:
        process_env = bindgen_environment(process_env)
    uv = shutil.which("uv")
    if uv is None:
        user_uv = Path.home() / ".local" / "bin" / "uv"
        if not user_uv.is_file():
            raise FileNotFoundError("uv is not on PATH or installed at ~/.local/bin/uv")
        uv = str(user_uv)
    subprocess.run(
        [uv, "sync", "--frozen", "--project", str(project)],
        env=process_env,
        check=True,
    )

    output = Path(os.environ["POMSET_OUTPUT_DIR"]) / "receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "synchronized",
                "project": str(project),
                "environment": str(environment),
                "cache": str(cache / "uv"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
