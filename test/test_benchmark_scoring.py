"""Scoring must reload actual checkpoints and align targets by name."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from electrolyte_fm.models.lm_finetuning import LMFinetuning
from opt.benchmarks.scripts.checkpoint_selection import export_model
from test.test_feature_miner_workflow import make_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("released", [False, True])
def test_scoring_loads_portable_model_and_preserves_channel_order(tmp_path, released):
    snapshot = tmp_path / "base"
    make_snapshot(snapshot)
    model = LMFinetuning(
        output_size=2,
        encoder_ckpt=str(snapshot),
        tokenizer=str(snapshot),
        task="regression",
        target_columns=["beta", "alpha"],
        metrics=["mae"],
        transform="standardize",
        track_oov=False,
    )
    model.configure_model()
    for parameter in model.task_network.parameters():
        parameter.data.zero_()
    model.transform.mean.copy_(torch.tensor([0.75, 0.25]))
    model.transform.std.fill_(1)
    checkpoint = tmp_path / "head.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    packaged = tmp_path / "model"
    export_model(checkpoint, snapshot, packaged)
    # Inference must work without the training checkpoint or the original cache.
    checkpoint.unlink()
    shutil.rmtree(snapshot)
    cell, registry, definitions, molecules = [
        tmp_path / name
        for name in (
            "cell.json",
            "registry.yaml",
            "tasks.jsonnet",
            "molecules.csv",
        )
    ]
    cell.write_text(
        json.dumps(
            {
                "test_data_used": False,
                "selection_split": "validation",
                "validation_loss": 0.25,
                "learning_rate": 0.001,
                "checkpoint": str(checkpoint),
            }
        )
    )
    registry.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "key": "tiny",
                        "base": "tiny",
                        "dataset": "synthetic",
                        "targets": ["alpha", "beta"],
                        "task": "regression",
                        "release": {
                            "id": "example/tiny",
                            "revision": "pinned",
                            "validation_value": 0.25,
                        }
                        if released
                        else None,
                    }
                ]
            }
        )
    )
    definitions.write_text("{}")
    molecules.write_text(
        "smiles,alpha,beta\nc1ccccc1,0.25,0.75\nCCO,0.25,0.75\ninvalid,0.25,0.75\n"
    )
    output = tmp_path / "scored"
    subprocess.run(
        [
            sys.executable,
            "-m", "opt.benchmarks.scripts.score_test",
            *(
                ["--model", str(packaged)]
                if released
                else ["--cells", str(cell), "--models", str(packaged)]
            ),
            "--registry",
            str(registry),
            "--task-definitions",
            str(definitions),
            "--case",
            "tiny",
            "--test-csv",
            str(molecules),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
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
    result = json.loads((output / "metrics.json").read_text())
    assert result["metrics"] == {"mae": 0, "rmse": 0}
    assert result["input_rows"] == 3
    assert result["predicted_rows"] == 2
    assert result["encoding"] == "smiles-kekule"
    assert result["split"] == "test"
    assert json.loads((output / "selection.json").read_text())["test_used"] is False
    assert (output / "model").is_symlink()
    assert (output / "model/config.json").is_file()
