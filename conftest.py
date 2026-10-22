"""Shared CLI-based workflow fixtures for tests under both test/ and opt/."""

import functools
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def copy_workflow_definitions():
    def copy(workspace):
        sources = [
            ROOT / "workflow.pomset",
            ROOT / "config.pomset",
            ROOT / "opt/benchmarks/registry.yaml",
            ROOT / "opt/interp_embeddings/compas.json",
            ROOT / "opt/screening/plots/Project.toml",
            *ROOT.glob("opt/*/*.pomset"),
            *ROOT.glob("opt/screening/campaigns/*.json"),
        ]
        for source in sources:
            target = workspace / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (workspace / ".pomset").mkdir(exist_ok=True)

    return copy


@pytest.fixture(scope="session")
def inspect_workflow(copy_workflow_definitions):
    @functools.cache
    def inspect(workflow="workflow.pomset", pattern="*"):
        with TemporaryDirectory(prefix="mist-pomset-test-") as temporary:
            workspace = Path(temporary)
            copy_workflow_definitions(workspace)
            output = subprocess.check_output(
                ["pomset", "task", "show", pattern, "--workflow", workflow, "--json"],
                cwd=workspace,
                text=True,
                timeout=30,
            )
            records = json.loads(output)
            if isinstance(records, dict):
                records = [records]
            return [record["logical"] for record in records]

    return inspect
