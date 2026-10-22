"""Validate probe fitting against current and released Lipinski schemas."""

import ast
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from datasets import Dataset, DatasetDict
from smirk import SmirkTokenizerFast
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel

from electrolyte_fm.data_modules.lipinski_dataset import LipinskiDataModule
from electrolyte_fm.models.linear_probe import LightningProbe, per_layer_probe
from opt.FeatureMiner.submit_linear_probes import probe_config

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "lipinski_h_donor",
    "lipinski_h_acceptor",
    "lipinski_mwt",
    "lipinski_log_p",
    "lipinski",
]


def make_data(path, legacy):
    rows = [
        {
            "smi": smi,
            **{
                name.replace("lipinski", "lipinki") if legacy else name: bool(
                    (i + j) % 2
                )
                for j, name in enumerate(TARGETS)
            },
        }
        for i, smi in enumerate(["CCO", "O", "CCN", "CO", "CC", "CCC", "CCCC", "CN"])
    ]
    DatasetDict(
        {split: Dataset.from_list(rows) for split in ("train", "validation")}
    ).save_to_disk(path / "data")


def make_snapshot(path):
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    tokenizer.save_pretrained(path)
    RobertaForMaskedLM(
        RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=2048,
        )
    ).save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("legacy", [False, True])
def test_lipinski_release_labels_and_validation_only_splits(tmp_path, legacy):
    make_data(tmp_path, legacy)
    make_snapshot(tmp_path / "snapshot")
    data = LipinskiDataModule(
        str(tmp_path), tokenizer=str(tmp_path / "snapshot"), num_workers=0
    )
    assert set(data.dataset) == {"train", "validation"}
    assert all(name in data.dataset["train"].column_names for name in TARGETS)
    data.setup("fit")
    batch = next(iter(data.val_dataloader()))
    assert torch.equal(
        batch["target"],
        torch.tensor([[bool((i + j) % 2) for j in range(5)] for i in range(8)]),
    )
    assert data.test_dataset is None


@pytest.mark.parametrize(("hidden_size", "batch_size"), [(512, 128), (2048, 8)])
def test_probe_fit_defaults_to_paper_step_budget(tmp_path, hidden_size, batch_size):
    snapshot = tmp_path / "snapshot"
    RobertaConfig(hidden_size=hidden_size).save_pretrained(snapshot)
    config = probe_config(snapshot, None, tmp_path, tmp_path / "fit")
    assert config["trainer"]["max_steps"] == 10_000
    assert config["trainer"]["max_epochs"] == -1
    assert config["trainer"]["max_time"] == "00:07:30:00"
    assert config["model"]["probes"] is None  # Dimensions come from the encoder.
    assert config["model"]["model"]["init_args"]["encoder"] == str(snapshot)
    assert config["model"]["optimizer"]["init_args"]["lr"] == 0.001
    assert config["model"]["lr_schedule"] is None
    assert config["data"]["init_args"]["batch_size"] == batch_size
    with pytest.raises(ValueError, match="positive"):
        probe_config(snapshot, None, tmp_path, tmp_path / "fit", batch_size=0)


def test_workflow_uses_existing_fit_script_without_jsonnet_or_probe_cli():
    tree = ast.parse((ROOT / "opt/FeatureMiner/workflow.pomset").read_text())
    literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert literals.count("./submit_linear_probes.py") == 2
    assert "../../submit/linear_probe.jsonnet" not in literals
    assert "scripts/train_probes.py" not in literals
    assert "probes.json" not in literals
    source = (ROOT / "electrolyte_fm/models/linear_probe.py").read_text()
    assert "class ProbeCLI" not in source
    assert "cli = MistLightningCLI(" in source
    adapter = (ROOT / "opt/FeatureMiner/submit_linear_probes.py").read_text()
    assert "linear_probe.jsonnet" not in adapter
    assert "cli_main(probe_config(" in adapter


@pytest.mark.parametrize("hidden_size,layers", [(8, 1), (16, 2)])
def test_probe_dimensions_follow_encoder_and_only_probes_train(hidden_size, layers):
    encoder = RobertaModel(
        RobertaConfig(
            vocab_size=20,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=layers,
            num_attention_heads=2,
        )
    )
    original = encoder.embeddings.word_embeddings.weight.detach().clone()
    model = LightningProbe(encoder)
    probes = dict(model.named_probes())
    assert len(probes) == 4 * layers
    for name, probe in probes.items():
        assert probe.out_features == 5
        assert probe.in_features == hidden_size * (
            2 if name.endswith("intermediate") else 1
        )
    optimizer = model.configure_optimizers()
    original_probe = model._probes[0].weight.detach().clone()
    model.forward_fit(
        {
            "input_ids": torch.tensor([[0, 5, 2, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "target": torch.tensor([[0, 1, 0, 1, 0]]),
        }
    )["loss"].backward()
    optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert torch.equal(encoder.embeddings.word_embeddings.weight, original)
    assert not torch.equal(model._probes[0].weight, original_probe)

    legacy = LightningProbe(
        encoder,
        probes=partial(
            per_layer_probe, hidden_size=hidden_size, features=5, n_layers=layers
        ),
    )
    assert len(dict(legacy.named_probes())) == layers


def make_selected_model(path, snapshot, kind):
    from electrolyte_fm.models.lm_finetuning import LMFinetuning
    from electrolyte_fm.models.prod_finetune import MISTFinetuned
    from electrolyte_fm.utils.tokenizer import load_tokenizer

    trained = LMFinetuning(
        output_size=1,
        encoder_ckpt=str(snapshot),
        tokenizer=str(snapshot),
        task="regression",
        target_columns=["value"],
        metrics=["mae"],
        track_oov=False,
    )
    trained.configure_model()
    if kind == "lightning":
        checkpoint = path / "selected.ckpt"
        torch.save(
            {
                "state_dict": trained.state_dict(),
                "hyper_parameters": dict(trained.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            checkpoint,
        )
        from opt.benchmarks.scripts.checkpoint_selection import export_model

        output = path / "exported"
        export_model(checkpoint, snapshot, output)
        checkpoint.unlink()
        return output
    else:
        checkpoint = path / "released"
        tokenizer = load_tokenizer(str(snapshot))
        model = MISTFinetuned.from_components(
            trained.encoder,
            trained.task_network,
            trained.transform,
            tokenizer=tokenizer,
            channels=["value"],
            copy=False,
        )
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        return checkpoint


@pytest.mark.parametrize("source", ["pretrained", "release", "lightning"])
@pytest.mark.parametrize(
    ("budget_args", "completed_steps", "requested_steps", "reached"),
    [
        (["--max-epochs", "2"], 4, 10_000, False),
        (["--max-steps", "3"], 3, 3, True),
    ],
)
def test_existing_script_fits_and_saves_every_hook(
    tmp_path, budget_args, completed_steps, requested_steps, reached, source
):
    make_data(tmp_path, legacy=True)
    make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "probes"
    model_args = []
    if source != "pretrained":
        model = make_selected_model(tmp_path, tmp_path / "snapshot", source)
        model_args = ["--model", str(model)]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "opt/FeatureMiner/submit_linear_probes.py"),
            *model_args,
            "--base-model",
            str(tmp_path / "snapshot"),
            "--data",
            str(tmp_path),
            "--output-dir",
            str(output),
            *budget_args,
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ],
        check=True,
        timeout=120,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    summary = torch.load(output / "fit.pt", weights_only=True)
    assert summary == {
        "training_steps": completed_steps,
        "requested_steps": requested_steps,
        "step_budget_reached": reached,
    }
    assert not (output / "probes.json").exists()
    paths = list((output / "checkpoints").glob("*.ckpt"))
    assert len(paths) == 4
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert not any(key.startswith("model.") for key in checkpoint["state_dict"])
        assert not isinstance(
            checkpoint["hyper_parameters"].get("model"), torch.nn.Module
        )
        model_args = checkpoint["hyper_parameters"]["model"]["init_args"]
        expected_model = (
            str(tmp_path / "snapshot")
            if source == "pretrained"
            else str(tmp_path / "exported")
            if source == "lightning"
            else str(tmp_path / "released")
        )
        assert (
            model_args.get("encoder", model_args.get("name_or_path")) == expected_model
        )
        assert 0 < checkpoint["global_step"] <= completed_steps
        assert checkpoint["hyper_parameters"]["optimizer"]["init_args"]["lr"] == 0.001
        assert checkpoint["hyper_parameters"]["lr_schedule"] is None
        for index, name in enumerate(checkpoint["state_dict"]["_prob_points"]):
            assert checkpoint["state_dict"][f"_probes.{index}.weight"].shape == (
                5,
                16 if name.endswith("intermediate") else 8,
            )
            assert 0 <= checkpoint["probe_metrics"][f"val/{name}-auroc"] <= 1
            assert checkpoint["probe_metrics"][f"val/{name}-loss"] >= 0


@pytest.mark.parametrize("kind", ["csv", "wandb", "disabled"])
def test_shared_cli_accepts_non_wandb_loggers_and_preserves_tags(tmp_path, kind):
    from electrolyte_fm.utils.cli import MistLightningCLI

    logger = (
        False
        if kind == "disabled"
        else {
            "class_path": f"lightning.pytorch.loggers.{'WandbLogger' if kind == 'wandb' else 'CSVLogger'}",
            "init_args": {"save_dir": str(tmp_path)},
        }
    )
    cli = MistLightningCLI(
        pl.LightningModule,
        args={
            "tags": ["probe-test"],
            "trainer": {"logger": logger, "accelerator": "cpu"},
        },
        run=False,
        save_config_callback=None,
    )
    if kind == "wandb":
        assert cli.trainer.logger._wandb_init["tags"] == ["probe-test"]
    elif kind == "disabled":
        assert cli.trainer.logger is None
    else:
        assert type(cli.trainer.logger).__name__ == "CSVLogger"
