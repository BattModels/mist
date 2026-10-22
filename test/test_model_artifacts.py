"""Model artifacts remain usable when staged, retried, and exported."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from electrolyte_fm.utils import stage_hf

ROOT = Path(__file__).resolve().parents[1]


def snapshot_in(cache, revision="pinned-revision"):
    snapshot = cache / "models--example--model/snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "roberta"}')
    return snapshot


def test_model_link_retries_replace_links_and_preserve_model_data(tmp_path):
    first = snapshot_in(tmp_path / "cache", "first")
    second = snapshot_in(tmp_path / "cache", "second")
    output = tmp_path / "artifacts/model"
    stage_hf.link_model(first, output)
    assert output.is_symlink()
    assert not Path(os.readlink(output)).is_absolute()
    original_inode = output.lstat().st_ino
    stage_hf.link_model(first, output)
    assert output.lstat().st_ino == original_inode
    stage_hf.link_model(second, output)
    assert output.resolve() == second
    assert (first / "config.json").is_file()
    shutil.rmtree(second)
    stage_hf.link_model(first, output)
    assert output.resolve() == first
    assert sorted(path.name for path in output.parent.iterdir()) == ["model"]


def test_model_link_survives_symlinked_output_ancestor_and_reruns(tmp_path):
    snapshot = snapshot_in(tmp_path / "cache")
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch/jobs/results"
    checkout.mkdir()
    scratch.mkdir(parents=True)
    (checkout / "results").symlink_to(scratch, target_is_directory=True)
    output = checkout / "results/stage/model"
    stage_hf.link_model(snapshot, output)
    stage_hf.link_model(snapshot, output)
    assert output.resolve(strict=True) == snapshot
    assert (output / "config.json").read_text() == (
        snapshot / "config.json"
    ).read_text()
    assert (scratch / "stage/model").is_symlink()


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
@pytest.mark.parametrize("link_files", [False, True])
def test_model_link_preserves_existing_artifacts(tmp_path, existing_kind, link_files):
    snapshot = snapshot_in(tmp_path / "cache")
    output = tmp_path / "model"
    if existing_kind == "directory":
        output.mkdir()
        artifact = output / "weights"
    else:
        artifact = output
    artifact.write_text("already imported model")
    with pytest.raises(FileExistsError, match="existing artifact"):
        stage_hf.link_model(snapshot, output, link_files=link_files)
    assert artifact.read_text() == "already imported model"


def test_model_file_links_migrate_retry_and_replace_without_stale_files(tmp_path):
    first = snapshot_in(tmp_path / "cache", "first")
    second = snapshot_in(tmp_path / "cache", "second")
    (first / "old-weights.bin").write_bytes(b"first weights")
    output = tmp_path / "test/model"
    stage_hf.link_model(first, output)
    stage_hf.link_model(first, output, link_files=True)
    assert output.is_dir() and not output.is_symlink()
    assert all(path.is_symlink() for path in output.iterdir())
    original_inode = output.stat().st_ino
    stage_hf.link_model(first, output, link_files=True)
    assert output.stat().st_ino == original_inode
    stage_hf.link_model(second, output, link_files=True)
    assert (output / "config.json").resolve() == second / "config.json"
    assert not (output / "old-weights.bin").exists()
    assert (first / "old-weights.bin").read_bytes() == b"first weights"


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
def test_model_file_links_export_with_array_producer(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    workflow = '''sweep = task(
    name="sweep", cmd=["true"], executor=array(items=["a"]),
    output_dir=Path("sweep") / array.item, outputs={"model": Path("model")},
    validator=validate_checksums(follow_symlinks=True),
)
task(
    name="test", cmd=["true"], executor=shell(), inputs=sweep.outputs.model,
    output_dir=Path("test"), outputs={"model": Path("model")},
    validator=validate_checksums(follow_symlinks=True),
)
'''
    for workspace in (source, destination):
        workspace.mkdir()
        (workspace / ".pomset").mkdir()
        (workspace / "workflow.pomset").write_text(workflow)
    model = source / "sweep/a/model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type": "example"}')
    (model / "weights.safetensors").write_bytes(b"checkpoint weights")
    stage_hf.link_model(model, source / "test/model", link_files=True)

    def pomset(workspace, *args):
        return subprocess.check_output(
            ["pomset", *args, "--pomset-dir", str(workspace / ".pomset")],
            cwd=workspace, stderr=subprocess.PIPE, timeout=30,
        )

    pomset(source, "task", "import", "--all")
    bundle = tmp_path / "models.tar"
    bundle.write_bytes(pomset(source, "export", "--with-files"))
    shutil.rmtree(source)
    pomset(destination, "import", str(bundle))
    for relative in ("sweep/a/model", "test/model"):
        restored = destination / relative
        assert (restored / "weights.safetensors").read_bytes() == b"checkpoint weights"
        assert not (restored / "weights.safetensors").is_symlink()


def run_stage(monkeypatch, snapshot, output, *extra, revision="pinned-revision"):
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(snapshot)

    monkeypatch.setattr(stage_hf, "snapshot_download", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_hf.py",
            "example/model",
            "--revision",
            revision,
            "--output",
            str(output),
            *extra,
        ],
    )
    stage_hf.main()
    return calls


def test_stage_keeps_pinned_download_filters_offline_and_cache_refs(
    tmp_path, monkeypatch
):
    snapshot = snapshot_in(tmp_path / "cache")
    output = tmp_path / "model"
    calls = run_stage(
        monkeypatch,
        snapshot,
        output,
        "--offline",
        "--allow-pattern",
        "*.json",
        "--allow-pattern",
        "*.safetensors",
        "--cache-ref",
        "pr/7",
    )
    assert calls == [
        (
            "example/model",
            {
                "revision": "pinned-revision",
                "cache_dir": str(stage_hf.CACHE / "huggingface/hub"),
                "local_files_only": True,
                "allow_patterns": ["*.json", "*.safetensors"],
            },
        )
    ]
    assert (snapshot.parents[1] / "refs/pr/7").read_text() == "pinned-revision"
    assert output.resolve() == snapshot


@pytest.mark.parametrize("failure", ["revision", "empty", "cache-ref"])
def test_stage_rejects_invalid_snapshot_before_publishing(
    tmp_path, monkeypatch, failure
):
    snapshot = snapshot_in(tmp_path / "cache")
    output = tmp_path / "model"
    options = {}
    extra = []
    if failure == "revision":
        options["revision"] = "different-revision"
        message = "resolved revision"
    elif failure == "empty":
        (snapshot / "config.json").unlink()
        message = "contains no files"
    else:
        extra = ["--cache-ref", "../outside"]
        message = "relative cache ref"
    with pytest.raises(ValueError, match=message):
        run_stage(monkeypatch, snapshot, output, *extra, **options)
    assert not output.exists()
    assert not output.is_symlink()


def generate_sweep(output):
    source = ROOT / "opt/screening/electrolyte_sweep.py"
    # The screening project pins Python 3.10 for its pyDOE2 dependency.
    python = ROOT / "opt/screening/.venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    subprocess.run(
        [str(python), str(source), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(output.read_text())


def test_screening_sweep_is_a_dict_of_unnamed_parameter_sets(tmp_path):
    document = generate_sweep(tmp_path / "campaigns/performance.json")
    assert set(document) == {"runs"}
    runs = document["runs"]
    assert runs
    assert [run["gpus"] for run in runs] == sorted(run["gpus"] for run in runs)
    for run in runs:
        assert set(run) == {
            "gpus",
            "batch_size",
            "n_fragments",
            "epoch_size",
            "limit_db_fragments",
            "seconds",
        }
        assert run["epoch_size"] in {
            int(run["n_fragments"] * fraction) for fraction in (0.9, 1, 1.1)
        }
        assert run["seconds"] == 300


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
def test_real_workflow_fingerprints_survive_workspace_relocation(copy_workflow_definitions, tmp_path):
    graphs = []
    for name in ("original", "relocated"):
        workspace = tmp_path / name
        copy_workflow_definitions(workspace)
        result = subprocess.run(
            ["pomset", "task", "show", "*", "--json"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        graphs.append({record["logical"]["id"]: record["logical"]
                       for record in json.loads(result.stdout)})
    original, relocated = graphs
    assert original.keys() == relocated.keys()
    for name in original:
        for key in ("execution_fingerprint", "contract_fingerprint"):
            assert original[name][key] == relocated[name][key], (name, key)
    # Cover the artifact-producing stage, trained models, and serialized
    # screening configuration rather than only a synthetic example workflow.
    assert {
        "benchmark-stage-mist-28m",
        "benchmark-stage-release-mist-28m-bp",
        "benchmark-mist-1p8b-bp-lr-sweep-bounded",
        "benchmark-mist-1p8b-bp-test",
        "screening-mist-28m-electrolyte-config",
        "screening-mist-28m-electrolyte",
    } <= original.keys()


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
def test_generated_sweep_expands_workflow_and_retains_artifact_inputs(copy_workflow_definitions, tmp_path):
    copy_workflow_definitions(tmp_path)
    runs = generate_sweep(tmp_path / "opt/screening/campaigns/performance.json")["runs"]
    result = subprocess.run(
        ["pomset", "task", "show", "screening-*", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    graph = {record["logical"]["id"]: record["logical"]
             for record in json.loads(result.stdout)}
    for family in ("mist-28m", "mist-1p8b"):
        prefix = f"screening-{family}-sweep-"
        assert {name for name in graph if name.startswith(prefix)} == {
            prefix + str(index) + suffix
            for index in range(len(runs))
            for suffix in ("", "-config")
        }
        scaling = graph[f"screening-{family}-scaling-plots"]
        for index, settings in enumerate(runs):
            name = prefix + str(index)
            configured = graph[name + "-config"]
            screened = graph[name]
            config = ast.literal_eval(configured["cmd"][3]["Literal"])
            for key in (
                "batch_size",
                "n_fragments",
                "epoch_size",
                "limit_db_fragments",
            ):
                assert config["generation"][key] == settings[key]
            assert config["limit_walltime"] == settings["seconds"]
            assert {
                "producer": "screening-generate-performance-sweep",
                "key": {"Named": "settings"},
            } in configured["inputs"]
            for item in configured["inputs"]:
                if item["key"] == {"Named": "model"}:
                    assert item in screened["inputs"]
            resources = screened["resources"]
            assert resources["nodes"] * resources["gpus_per_node"] == settings["gpus"]
            for output in ("config", "trace"):
                assert {"producer": name, "key": {"Named": output}} in scaling["inputs"]


@pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
def test_pomset_export_import_bundles_directory_and_nested_blob_links(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModel,
        AutoTokenizer,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaModel,
    )

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    cache = tmp_path / "external-hf-cache"
    snapshot = snapshot_in(cache)
    blob_dir = snapshot.parents[1] / "blobs"
    blob_dir.mkdir()
    original = RobertaModel(
        RobertaConfig(
            vocab_size=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "C": 1, "O": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    seed = tmp_path / "seed-model"
    original.save_pretrained(seed)
    tokenizer.save_pretrained(seed)
    payloads = {path.name: path.read_bytes() for path in seed.iterdir()}
    payloads["nested/generation_config.json"] = b'{"max_length": 20}'
    (snapshot / "config.json").unlink()
    for index, (name, payload) in enumerate(payloads.items()):
        blob = blob_dir / str(index)
        blob.write_bytes(payload)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    stage_hf.link_model(snapshot, source / "out/model")
    workflow = """task(
    name="staged-model",
    cmd=["true"],
    executor=shell(),
    output_dir=Path("out"),
    outputs={"model": Path("model")},
    validator=validate_checksums(follow_symlinks=True),
)
"""
    for workspace in (source, destination):
        (workspace / "workflow.pomset").write_text(workflow)
        (workspace / ".pomset").mkdir()

    def pomset(workspace, *args, binary=False):
        result = subprocess.run(
            ["pomset", *args, "--pomset-dir", str(workspace / ".pomset")],
            cwd=workspace,
            capture_output=True,
            text=not binary,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    pomset(source, "submit", "--backend", "local", "--no-progress")
    bundle = tmp_path / "model.tar"
    bundle.write_bytes(pomset(source, "export", "--with-files", binary=True))
    shutil.rmtree(cache)
    shutil.rmtree(source)
    shutil.rmtree(seed)
    pomset(destination, "import", str(bundle))
    model = destination / "out/model"
    assert model.is_dir()
    assert not model.is_symlink()
    for name, payload in payloads.items():
        restored = model / name
        assert restored.read_bytes() == payload
        assert not restored.is_symlink()
    restored_model = AutoModel.from_pretrained(model, local_files_only=True)
    restored_tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    with torch.no_grad():
        expected = original(**tokenizer("O", return_tensors="pt")).last_hidden_state
        actual = restored_model(
            **restored_tokenizer("O", return_tensors="pt")
        ).last_hidden_state
    torch.testing.assert_close(actual, expected)
