"""Plotting entrypoints stay executable CLIs, including PythonCall's uv launch."""

import json
import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    "opt/design/plots.jl",
    "opt/FeatureMiner/plots/lipinski_probes.jl",
    "opt/synth_access/plot.jl",
    "opt/sterochemistry/plot.jl",
    "opt/screening/plots/plots.jl",
    "opt/screening/plots/plot_lio2.jl",
    "opt/screening/plots/supplementary.jl",
]


@pytest.mark.parametrize("relative", ENTRYPOINTS)
def test_direct_cli_uses_shebang_and_preserves_arguments(tmp_path, relative):
    script = ROOT / relative
    source = script.read_text()
    assert source.startswith("#!/usr/bin/env -S ")
    interpreter = shlex.split(source.splitlines()[0].removeprefix("#!/usr/bin/env -S "))
    assert interpreter[0] in {"uv", "julia"}
    assert "--project=@script" in interpreter
    assert "--startup-file=no" in interpreter
    assert script.stat().st_mode & stat.S_IXUSR

    # Capture the interpreter argv without importing Julia packages, accessing
    # the network, or loading any of the production checkpoints.
    stub = tmp_path / interpreter[0]
    stub.write_text(
        "#!/usr/bin/python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [str(script), "--output-dir", "path with spaces", "--base", "mist-1p8b"],
        check=True,
        capture_output=True,
        text=True,
        cwd=script.parent,
        env={**os.environ, "PATH": str(tmp_path)},
    )
    assert json.loads(result.stdout) == [
        *interpreter[1:],
        str(script),
        "--output-dir",
        "path with spaces",
        "--base",
        "mist-1p8b",
    ]


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
@pytest.mark.parametrize("relative", ENTRYPOINTS)
def test_workflow_tracks_plotting_entrypoint(inspect_workflow, relative):
    script = ROOT / relative
    for task in inspect_workflow():
        for argument in task["cmd"]:
            path = argument.get("Path")
            if path is None or path["path"]["kind"] != "static":
                continue
            value = path["path"]["value"]
            resolved = (ROOT / value[2:] if value.startswith("//") else
                        ROOT / path["origin"]["value"] / value)
            if resolved.resolve() == script:
                return
    pytest.fail(f"No task command tracks {relative}")
