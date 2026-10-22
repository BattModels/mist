"""Seed uncertainty selects by validation and retains losses without weights."""

import json
import math
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test.test_feature_miner_workflow import make_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seeds(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from opt.benchmarks.scripts import moleculenet_seeds

    return moleculenet_seeds


def test_best_validation_weights_are_copied_in_memory(seeds):
    model = torch.nn.Linear(1, 1, bias=False)
    best = seeds.BestValidationWeights()
    trainer = SimpleNamespace(
        sanity_checking=True,
        state=SimpleNamespace(fn="fit"),
        global_step=1,
        callback_metrics={"val/loss_epoch": 0.1},
    )
    best.on_validation_end(trainer, model)
    assert best.weights is None
    trainer.sanity_checking = False
    model.weight.data.fill_(1)
    best.on_validation_end(trainer, model)
    model.weight.data.fill_(2)
    trainer.callback_metrics["val/loss_epoch"] = 0.2
    best.on_validation_end(trainer, model)
    assert best.loss == 0.1
    assert best.weights["weight"].item() == 1
    trainer.callback_metrics["val/loss_epoch"] = float("nan")
    best.on_validation_end(trainer, model)
    assert best.weights["weight"].item() == 1


@pytest.mark.parametrize(
    ("task", "as_cli"),
    [("regression", False), ("binary", False), pytest.param("regression", True, id="script")],
)
def test_fit_seed_publishes_losses_without_checkpoint_or_model(
    seeds, tmp_path, monkeypatch, task, as_cli
):
    base, split_dir = tmp_path / "base", tmp_path / "data"
    make_snapshot(base)
    split_dir.mkdir()
    for name in ("train", "validation", "test"):
        (split_dir / f"{name}.csv").write_text(
            "smiles,value\nCCO,0\nCCN,1\nCCC,0\nCCCC,1\n"
        )
    selection = tmp_path / "selection.json"
    recipe = {
        "learning_rate": 0.001, "betas": [0.87, 0.997], "warmup_steps": 0,
        "freeze_encoder": True, "encoding": "smiles", "randomize": True,
        "max_steps": 2, "max_time": "00:00:05:00", "val_check_interval": 2,
        "batch_size": 2, "effective_batch_size": 4,
    }
    selection.write_text(
        json.dumps(
            {
                "task": "tiny",
                "seed": 42,
                "learning_rate": 0.001,
                "selection_split": "validation",
                "test_data_used": False,
                **({"training": recipe} if as_cli else {}),
            }
        )
    )
    args = Namespace(
        selection=selection,
        task="tiny",
        seed=43,
        model=base,
        dataset=split_dir,
        output_dir=tmp_path / "fit",
        batch_size=2,
        effective_batch_size=4,
        max_steps=2,
        warmup_steps=0,
        max_time="00:00:05:00",
        val_check_interval=2,
        num_workers=0,
        test_csv=split_dir / "test.csv",
    )
    case = {
        "task": task,
        "targets": ["value"],
        "base": "tiny-base",
        "dataset": "tiny",
        "metrics": ["auroc"] if task == "binary" else ["mae"],
    }
    original = seeds.SplitCSVDataModule._get_dataset
    test_reads = []

    def checked_load(data):
        if data.test_csv is not None:
            assert data.trainer.state.fn != "fit", (
                "Test data was opened during selection"
            )
            test_reads.append(data.test_csv)
        return original(data)

    def forbidden_checkpoint(*args, **kwargs):
        raise AssertionError("Metrics-only fitting attempted to write a checkpoint")

    monkeypatch.setattr(seeds.SplitCSVDataModule, "_get_dataset", checked_load)
    monkeypatch.setattr(seeds.Trainer, "save_checkpoint", forbidden_checkpoint)
    if as_cli:
        registry, definitions = tmp_path / "registry.json", tmp_path / "tasks.jsonnet"
        registry.write_text(json.dumps({"cases": [{"key": args.task, **case}]}))
        definitions.write_text("{}")
        command = [
            sys.executable, "-m", "opt.benchmarks.scripts.moleculenet_seeds", "fit",
            "--registry", str(registry), "--task-definitions", str(definitions),
        ]
        for key, value in vars(args).items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            command, cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        seeds.fit(args, case)
        assert test_reads
    result = json.loads((args.output_dir / "losses.json").read_text())
    assert result["seed"] == 43
    assert result["lr_selection_seed"] == 42
    assert result["completed_steps"] == 2
    assert result["learning_rate"] == 0.001
    assert all(math.isfinite(result[key]) for key in ("validation_loss", "test_loss"))
    assert not list(args.output_dir.rglob("*.ckpt"))
    assert not list(args.output_dir.rglob("*.safetensors"))
    assert not (args.output_dir / "model").exists()
    if as_cli:
        config = json.loads((args.output_dir / "config.json").read_text())
        assert result["training"] == recipe
        assert config["model"]["init_args"]["freeze_encoder"] is True
        assert config["model"]["init_args"]["optimizer"]["init_args"]["betas"] == recipe["betas"]
        assert config["data"]["init_args"]["encoding"] == "smiles"
        assert config["data"]["init_args"]["randomize"] is True


def test_seed_summary_requires_five_matching_runs(seeds, tmp_path):
    paths = []
    for index, seed in enumerate(range(42, 47), 1):
        path = tmp_path / f"{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "task": "tiny",
                    "base": "28m",
                    "dataset": "tiny",
                    "seed": seed,
                    "learning_rate": 0.001,
                    "validation_loss": index,
                    "test_loss": 2 * index,
                }
            )
        )
        paths.append(path)
    report = seeds.summarize(paths[::-1])
    assert report["n_seeds"] == 5
    assert report["summary"]["validation_loss"]["mean"] == 3
    assert report["summary"]["validation_loss"]["std"] == pytest.approx(math.sqrt(2.5))
    assert report["summary"]["test_loss"]["sem"] == pytest.approx(math.sqrt(2))
    with pytest.raises(ValueError, match="exactly one"):
        seeds.summarize(paths[:-1] + paths[:1])
    row = json.loads(paths[-1].read_text())
    row["learning_rate"] = 0.002
    paths[-1].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="learning_rate"):
        seeds.summarize(paths)
    row["learning_rate"] = 0.001
    row["training"] = {"freeze_encoder": True}
    paths[-1].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="training"):
        seeds.summarize(paths)


@pytest.mark.parametrize("cell_encoding", [None, "smiles"])
def test_seed42_evaluates_only_the_validation_winner(
    seeds, tmp_path, monkeypatch, cell_encoding
):
    from electrolyte_fm.models.prod_finetune import MISTFinetuned

    base, selected_model = tmp_path / "base", tmp_path / "selected-model"
    tokenizer = make_snapshot(base)
    module = seeds.LMFinetuning(
        output_size=1,
        encoder_ckpt=str(base),
        tokenizer=str(base),
        task="regression",
        metrics=["mae"],
        target_columns=["value"],
    )
    module.configure_model()
    module.transform.std.fill_(1)
    packaged = MISTFinetuned.from_components(
        module.encoder,
        module.task_network,
        module.transform,
        tokenizer,
        ["value"],
        copy=False,
    )
    packaged.config.encoding = "smiles"
    packaged.save_pretrained(selected_model)
    split_dir = tmp_path / "data"
    split_dir.mkdir()
    for name in ("train", "validation", "test"):
        (split_dir / f"{name}.csv").write_text("smiles,value\nCCO,0\nCCN,1\n")
    cells = []
    for index, loss in enumerate([0.4, 0.2]):
        path = tmp_path / f"cell-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "task": "tiny",
                    "validation_loss": loss,
                    "learning_rate": 0.001 * (index + 1),
                    **({"encoding": cell_encoding} if cell_encoding else {}),
                    "selection_split": "validation",
                    "test_data_used": False,
                    "betas": [0.87, 0.997],
                    "freeze_encoder": True,
                    "randomize": True,
                    "max_time": "00:07:45:00",
                }
            )
        )
        cells.append(path)
    args = Namespace(
        cells=cells,
        models=[tmp_path / "absent-loser", selected_model],
        dataset=split_dir,
        output_dir=tmp_path / "seed42",
        task="tiny",
        test_csv=split_dir / "test.csv",
        num_workers=0,
    )
    case = {
        "task": "regression",
        "targets": ["value"],
        "metrics": ["mae"],
        "base": "tiny-base",
        "dataset": "tiny",
    }
    original = seeds.SplitCSVDataModule._get_dataset

    def checked_load(data):
        selection = json.loads((args.output_dir / "selection.json").read_text())
        assert selection["learning_rate"] == 0.002
        assert selection["training"]["betas"] == [0.87, 0.997]
        assert selection["training"]["freeze_encoder"] is True
        assert selection["training"]["randomize"] is True
        assert selection["training"]["max_time"] == "00:07:45:00"
        assert data.encoding.value == "smiles"
        return original(data)

    monkeypatch.setattr(seeds.SplitCSVDataModule, "_get_dataset", checked_load)
    seeds.seed42(args, case)
    result = json.loads((args.output_dir / "losses.json").read_text())
    assert result["seed"] == 42
    assert result["selection_validation_loss"] == 0.2
    assert all(math.isfinite(result[key]) for key in ("validation_loss", "test_loss"))


def test_lr_choice_ignores_test_results_and_breaks_ties_by_lr():
    from opt.benchmarks.scripts.checkpoint_selection import select_lr

    cells = [
        {
            "validation_loss": val,
            "test_loss": test,
            "learning_rate": lr,
            "test_data_used": False,
            "selection_split": "validation",
        }
        for val, test, lr in [(0.2, 0.01, 0.001), (0.1, 0.3, 0.002), (0.1, 0.4, 0.001)]
    ]
    assert select_lr(cells) == 2
    cells[2]["test_data_used"] = True
    with pytest.raises(ValueError, match="validation-only"):
        select_lr(cells)
