"""Benchmark recipes preserve freezing, augmentation, and sealed evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from opt.benchmarks.scripts.split_csv_data import SplitCSVDataModule
from test.test_feature_miner_workflow import make_snapshot

ROOT = Path(__file__).resolve().parents[1]


def make_splits(path):
    path.mkdir()
    for name in ("train", "validation"):
        (path / f"{name}.csv").write_text(
            "smiles,value\nCCO,1.0\nCCN,2.0\nCCC,3.0\nCCCC,4.0\n"
        )


def test_muv_materializer_uses_current_dataset_api(tmp_path, monkeypatch):
    import pandas as pd

    from opt.benchmarks.scripts import materialize_moleculenet

    source = tmp_path / "source.csv"
    source.write_text(
        "smiles,activity\n" + "".join(f"{'C' * (i + 1)},{i % 2}\n" for i in range(20))
    )
    definitions = tmp_path / "tasks.jsonnet"
    definitions.write_text(
        json.dumps({"muv": {"split": "random", "target_columns": ["activity"]}})
    )
    output = tmp_path / "materialized"
    monkeypatch.setitem(materialize_moleculenet._URLS, "muv", source.as_uri())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_moleculenet.py",
            "--dataset",
            "muv",
            "--task-definitions",
            str(definitions),
            "--output-dir",
            str(output),
        ],
    )
    materialize_moleculenet.main()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["dataset"] == "muv"
    assert protocol["targets"] == ["activity"]
    splits = [pd.read_csv(output / f"{name}.csv") for name in protocol["counts"]]
    combined = pd.concat(splits)
    assert len(combined) == combined.smiles.nunique() == 20
    assert all(len(split) for split in splits)


def test_sweep_loader_does_not_require_or_read_test(tmp_path):
    snapshot, splits = tmp_path / "base", tmp_path / "splits"
    make_snapshot(snapshot)
    make_splits(splits)
    data = SplitCSVDataModule(
        str(splits), tokenizer=str(snapshot), target_columns=["value"]
    )
    assert set(data.dataset) == {"train", "validation"}
    # Even a present but unreadable test file must remain sealed during fitting.
    (splits / "test.csv").write_text('"malformed CSV without a closing quote')
    data.setup("fit")
    assert data.test_dataset is None
    assert not data.train_dataset.cache_files
    assert not data.val_dataset.cache_files
    assert next(iter(data.val_dataloader()))["target"].shape == (4, 1)
    with pytest.raises(RuntimeError, match="sealed"):
        data.test_dataloader()


@pytest.mark.parametrize("new_recipe,frozen", [(False, False), (True, False), (True, True)])
def test_sweep_cli_respects_recipe_and_reports_effective_batch(tmp_path, new_recipe, frozen):
    from transformers import AutoModel

    snapshot, splits = tmp_path / "base", tmp_path / "splits"
    make_snapshot(snapshot)
    make_splits(splits)
    original = AutoModel.from_pretrained(snapshot, add_pooling_layer=False)
    weight = original.embeddings.word_embeddings.weight.detach().clone()
    registry, definitions = [
        tmp_path / name for name in ("registry.yaml", "tasks.jsonnet")
    ]
    registry.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "key": "tiny",
                        "targets": ["value"],
                        "task": "regression",
                        "metrics": ["mae"],
                    }
                ]
            }
        )
    )
    definitions.write_text("{}")
    output = tmp_path / "fit"
    command = [
        sys.executable,
        "-m", "opt.benchmarks.scripts.train_screening_head",
        "--task",
        "tiny",
        "--registry",
        str(registry),
        "--task-definitions",
        str(definitions),
        "--dataset",
        str(splits),
        "--model",
        str(snapshot),
        "--learning-rate",
        "0.001",
        "--batch-size",
        "2",
        "--effective-batch-size",
        "4",
        "--max-steps",
        "2",
        # A two-step smoke run must leave the zero-LR warmup before its first
        # selected checkpoint, including when bf16 rounds validation losses.
        "--warmup-steps",
        "0",
        "--val-check-interval",
        "4",
        "--num-workers",
        "0",
        "--output-dir",
        str(output),
    ]
    if new_recipe:
        command.extend(["--betas", "0.87", "0.997", "--encoding", "smiles", "--randomize"])
        command[command.index("--warmup-steps") + 1] = "beta2"
    if frozen:
        command.append("--freeze-encoder")
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cell = json.loads((output / "cell.json").read_text())
    assert cell["microbatch_size"] == cell["gradient_accumulation"] == 2
    assert cell["effective_batch_size"] == 4
    assert cell["test_data_used"] is False
    assert cell["selection_split"] == "validation"
    assert cell["encoding"] == ("smiles" if new_recipe else "smiles-kekule")
    assert cell["randomize"] is new_recipe
    assert cell["freeze_encoder"] is frozen
    assert cell["validation_loss"] >= 0
    assert (output / "model/config.json").is_file()
    assert (output / "model/tokenizer_config.json").is_file()
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["model"]["init_args"]["freeze_encoder"] is frozen
    assert config["data"]["init_args"]["randomize"] is new_recipe
    checkpoint = torch.load(cell["checkpoint"], map_location="cpu", weights_only=False)
    expected_betas = [0.87, 0.997] if new_recipe else [0.9, 0.999]
    assert cell["betas"] == expected_betas
    assert list(checkpoint["optimizer_states"][0]["param_groups"][0]["betas"]) == expected_betas
    if new_recipe:
        assert cell["warmup_steps"] == "beta2"
        assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(
            0.001 * checkpoint["global_step"] / 667
        )
    fitted = checkpoint["state_dict"]["encoder.embeddings.word_embeddings.weight"]
    assert torch.equal(fitted, weight) is frozen
    # Simulate a scheduler retry. The logger/checkpoint directory must stay the
    # same, preserving the best score and avoiding another retained checkpoint
    # collection for every attempt.
    command[command.index("--max-steps") + 1] = "4"
    resumed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Restored all states" in resumed.stderr
    assert "dirpath has changed" not in resumed.stderr
    after = json.loads((output / "cell.json").read_text())
    assert after["validation_loss"] <= cell["validation_loss"]
    assert len(list((output / "metrics").glob("version_*"))) == 1
    # Aliases may point at the one best checkpoint; count physical files only.
    assert len({path.resolve() for path in output.rglob("*.ckpt")}) <= 2


def test_beta_sweeps_have_disjoint_outputs_and_validation_only_inputs(inspect_workflow):
    tasks = {
        item["id"]: item for item in inspect_workflow(
            pattern="benchmark-*-lr-sweep-bounded*"
        )
    }
    originals = {name: task for name, task in tasks.items() if name.endswith("bounded")}
    assert len(originals) == 40
    original_paths = {output["path"] for task in originals.values() for output in task["outputs"]}
    for name, original in originals.items():
        updated = tasks[name + "-betas-0p87-0p997"]
        assert updated["executor"] == original["executor"]
        assert len(updated["executor"]["Array"]["items"]) == 7
        assert updated["inputs"] == original["inputs"]
        assert updated["cmd"][-5:] == [
            {"Literal": arg} for arg in ["--betas", "0.87", "0.997", "--warmup-steps", "beta2"]
        ]
        for output in updated["outputs"]:
            assert "/sweep-betas-0p87-0p997/" in output["path"]
            assert output["path"] not in original_paths
        assert not any(artifact["key"].get("Named") == "test" for artifact in updated["inputs"])


def test_expanded_recipes_feed_one_validation_selection(inspect_workflow):
    tasks = {task["id"]: task for task in inspect_workflow(pattern="benchmark-*")}
    expanded = {name: task for name, task in tasks.items() if "-lr-sweep-bounded-randomized-" in name}
    assert len(expanded) == 24 * 2 * 2
    paths = []
    for name, task in expanded.items():
        items = task["executor"]["Array"]["items"]
        assert len(items) == 8
        assert "-lr8" in name
        rates = [float(item["value"]["value"]["value"]) for item in items]
        upper = 4e-3 if "-mist-28m-" in name else 1e-3
        assert rates == pytest.approx([1e-5 * (upper / 1e-5) ** (i / 7) for i in range(8)])
        args = [x["Literal"] for x in task["cmd"] if "Literal" in x]
        assert args[args.index("--max-time") + 1] == "00:07:45:00"
        assert args[args.index("--encoding") + 1] == "smiles"
        assert "--randomize" in args
        assert ("--freeze-encoder" in args) is ("-randomized-frozen" in name)
        assert not any(artifact["key"].get("Named") == "test" for artifact in task["inputs"])
        paths.extend(output["path"] for output in task["outputs"])
    assert len(paths) == len(set(paths))
    selections = [task for name, task in tasks.items() if name.endswith("-seed42-losses-expanded")]
    assert len(selections) == 24
    for task in selections:
        prefix = task["id"].removesuffix("-seed42-losses-expanded") + "-lr-sweep-bounded"
        expected = {name for name in tasks if name.startswith(prefix)}
        assert len(expected) == 6
        assert expected <= set(task["dependencies"])
        cells = [x for x in task["inputs"] if x["key"].get("ArrayElement", {}).get("name") == "cell"]
        models = [x for x in task["inputs"] if x["key"].get("ArrayElement", {}).get("name") == "model"]
        assert len(cells) == len(models) == 46
        assert [(x["producer"], x["key"]["ArrayElement"]["index"]) for x in cells] == [
            (x["producer"], x["key"]["ArrayElement"]["index"]) for x in models
        ]
        summary = tasks[task["id"].replace("-seed42-losses-expanded", "-seed-uncertainty-expanded")]
        assert summary["executor"] == "Job"
