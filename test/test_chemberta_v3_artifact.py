"""Cross-repository custom code travels with its model, without an HF cache."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from opt.sterochemistry.scripts.package_chemberta_v3 import CODE_REPOSITORY, package


def sources(root):
    model, code = root / "snapshot", root / "code"
    model.mkdir(parents=True)
    code.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "fixture", "hidden_size": 17,
        "auto_map": {"AutoConfig": CODE_REPOSITORY + "--configuration_fixture.FixtureConfig"},
    }))
    (model / "tokenizer_config.json").write_text(json.dumps({
        "auto_map": {"AutoTokenizer": [None, CODE_REPOSITORY + "--tokenization_fixture.FixtureTokenizer"]},
    }))
    (model / "weights.bin").write_bytes(b"unchanged fixture weights")
    (code / "configuration_fixture.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class FixtureConfig(PretrainedConfig):\n    model_type = 'fixture'\n"
    )
    (code / "tokenization_fixture.py").write_text("# Fixture tokenizer metadata only.\n")
    return model, code


def test_packaging_preserves_inputs_handles_reruns_and_rejects_unknown_code(tmp_path):
    model, code = sources(tmp_path)
    original = (model / "config.json").read_bytes()
    output = tmp_path / "output"
    package(model, code, output)
    (output / "stale.py").write_text("# Removed by a clean rerun.\n")
    package(model, code, output)
    assert not (output / "stale.py").exists()
    assert (model / "config.json").read_bytes() == original
    assert (output / "weights.bin").is_symlink()
    assert (output / "weights.bin").read_bytes() == (model / "weights.bin").read_bytes()
    assert json.loads((output / "tokenizer_config.json").read_text())["auto_map"] == {
        "AutoTokenizer": [None, "tokenization_fixture.FixtureTokenizer"],
    }
    with pytest.raises(ValueError, match="separate"):
        package(model, code, tmp_path)
    config = json.loads(original)
    config["auto_map"]["AutoConfig"] = "unknown/repository--config.Model"
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unstaged"):
        package(model, code, output)
    assert (output / "weights.bin").read_bytes() == b"unchanged fixture weights"


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
def test_current_pomset_import_loads_custom_config_without_source_cache(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    model, code = sources(tmp_path / "original-inputs")
    source.mkdir()
    destination.mkdir()
    package(model, code, source / "out/model")
    workflow = '''task(
    name="packaged-model", cmd=["true"], executor=shell(),
    output_dir=Path("out"), outputs={"model": Path("model")},
    validator=validate_checksums(follow_symlinks=True),
)
'''
    for workspace in (source, destination):
        (workspace / "workflow.pomset").write_text(workflow)
        (workspace / ".pomset").mkdir()

    def pomset(workspace, *args):
        return subprocess.run(
            ["pomset", *args], cwd=workspace, check=True,
            capture_output=True, timeout=30,
        ).stdout

    pomset(source, "submit", "--backend", "local", "--no-progress")
    bundle = tmp_path / "model.tar"
    bundle.write_bytes(pomset(source, "export", "--with-files"))
    shutil.rmtree(source)
    shutil.rmtree(model.parent)
    pomset(destination, "import", str(bundle))
    restored = destination / "out/model"
    assert (restored / "weights.bin").read_bytes() == b"unchanged fixture weights"
    assert not (restored / "weights.bin").is_symlink()
    environment = dict(
        os.environ, HF_HOME=str(tmp_path / "empty-hf-cache"),
        HF_MODULES_CACHE=str(tmp_path / "empty-module-cache"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    subprocess.run(
        [sys.executable, "-c",
         ("from transformers import AutoConfig; import sys; "
          "c=AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True); "
          "assert c.model_type == 'fixture' and c.hidden_size == 17"),
         str(restored)],
        env=environment, check=True, capture_output=True, timeout=30,
    )
