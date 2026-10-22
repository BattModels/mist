#!/usr/bin/env python3
"""Run and seal one validation-only tmQM learning-rate cell."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

from electrolyte_fm.utils.tokenizer import load_tokenizer
from submit.submit import jsonnet_load, merge_config


def completed_fit(output_dir: Path, steps: int) -> bool:
    """Return whether a previous attempt reached the requested terminal step."""
    versions = sorted(output_dir.glob("metrics/version_*"))
    if len(versions) != 1:
        return False
    metrics = versions[0] / "metrics.csv"
    if not metrics.is_file():
        return False
    observed = [
        int(float(row["step"]))
        for row in csv.DictReader(metrics.open())
        if row.get("step") and row.get("train/loss_step")
    ]
    return bool(observed) and max(observed) + 1 == steps


def checkpoint_state_file(checkpoint: Path) -> Path:
    """Resolve either a standard Lightning file or a DeepSpeed directory."""
    if checkpoint.is_file():
        return checkpoint.resolve()
    return checkpoint / "checkpoint/mp_rank_00_model_states.pt"


def resolve_fit_settings(model: str, snapshot: str | Path, batch_size: int) -> dict:
    """Derive runtime settings from the pinned model rather than the workflow."""
    config = AutoConfig.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    tokenizer = load_tokenizer(str(snapshot), local_files_only=True)
    identity = " ".join(
        str(value).lower()
        for value in (
            model,
            getattr(config, "model_type", ""),
            tokenizer.__class__.__name__,
        )
    )
    encoding = (
        "selfies" if "selformer" in identity or "selfies" in identity else "smiles"
    )

    limits = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 1 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    position_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and position_limit > 1:
        limits.append(position_limit - tokenizer.num_special_tokens_to_add(pair=False))
    if not limits:
        raise RuntimeError(f"model exposes no finite sequence limit: {model}")
    max_length = min(limits)

    declared_dtype = str(getattr(config, "torch_dtype", "")).lower()
    if "bfloat16" in declared_dtype and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    elif "float16" in declared_dtype and torch.cuda.is_available():
        precision = "16-mixed"
    elif (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not declared_dtype
    ):
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    return {
        "encoding": encoding,
        "max_sequence_length": max_length,
        "batch_size": batch_size,
        "val_batch_size": max(batch_size, 16),
        "precision": precision,
        "model_type": getattr(config, "model_type", None),
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100000)
    a = p.parse_args()

    snapshot = a.model_dir
    if not snapshot.is_dir():
        raise FileNotFoundError(f"staged model snapshot is unavailable: {snapshot}")
    if a.batch_size < 1:
        raise ValueError("batch size must be positive")
    settings = resolve_fit_settings(a.model, snapshot, a.batch_size)
    reuse_fit = completed_fit(a.output_dir, a.steps)
    if a.output_dir.exists() and not reuse_fit:
        if (a.output_dir / "cell.json").exists():
            raise RuntimeError(
                f"refusing to overwrite completed LR output: {a.output_dir}"
            )
        # A restartable Pomset attempt owns this output directory. Failed fits can
        # leave logger configuration and partial checkpoints that must not be
        # mistaken for files produced by the new attempt.
        shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    log = a.output_dir / "train.log"
    repository = Path(__file__).resolve().parents[3]
    submit_config = jsonnet_load(repository / "submit/tmQM.jsonnet")
    targets = submit_config["train"]["data"]["init_args"]["target_columns"]
    if submit_config["train"]["model"]["init_args"]["output_size"] != len(targets):
        raise RuntimeError("submit tmQM output-size/target contract disagrees")
    submit_config = merge_config(
        submit_config,
        {
            "train": {
                "tags": ["finetuning", "tmQM", a.model],
                "model": {
                    "init_args": {
                        "encoder_ckpt": str(snapshot),
                        "tokenizer": str(snapshot),
                        "freeze_encoder": False,
                        "target_columns": targets,
                        "optimizer": {"init_args": {"lr": a.learning_rate}},
                        "lr_schedule": {"init_args": {"num_training_steps": a.steps}},
                    }
                },
                "data": {
                    "class_path": "electrolyte_fm.data_modules.tmQMTrainValidationDataModule",
                    "init_args": {
                        "path": str(a.dataset),
                        "tokenizer": str(snapshot),
                        "target_columns": targets,
                        "encoding": settings["encoding"],
                        "truncation": True,
                        "batch_size": settings["batch_size"],
                        "val_batch_size": settings["val_batch_size"],
                        "max_seq_length": settings["max_sequence_length"],
                    },
                },
                "trainer": {
                    "max_steps": a.steps,
                    "max_time": "00:12:00:00",
                    "precision": settings["precision"],
                    "accelerator": "gpu",
                    "devices": 1,
                    "strategy": "auto",
                    "enable_progress_bar": False,
                    "val_check_interval": 1000,
                    "check_val_every_n_epoch": None,
                    "default_root_dir": str(a.output_dir),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": str(a.output_dir),
                            "name": "metrics",
                            "version": 0,
                        },
                    },
                },
            }
        },
    )
    config = a.output_dir / "submit-config.json"
    if not reuse_fit:
        config.write_text(json.dumps(submit_config["train"], indent=2) + "\n")
        train = repository / "train.py"
        command = [
            sys.executable,
            str(train),
            "fit",
            "--config",
            str(config),
        ]
        env = dict(
            os.environ,
            WANDB_MODE="disabled",
            MIST_CHECKPOINT_INTERVAL_MINUTES="10",
            JOB_CONFIG=json.dumps(submit_config),
        )
        with log.open("w") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False
            )
        if completed.returncode:
            raise RuntimeError(f"training exited {completed.returncode}; see {log}")

    versions = sorted(a.output_dir.glob("metrics/version_*"))
    if len(versions) != 1:
        raise RuntimeError("expected exactly one logger version")
    metrics = versions[0] / "metrics.csv"
    last_checkpoint = versions[0] / "checkpoints/last.ckpt"
    last_state = checkpoint_state_file(last_checkpoint)
    rows = list(csv.DictReader(metrics.open()))
    if any(
        value
        for row in rows
        for key, value in row.items()
        if key.startswith("test/") and value
    ):
        raise RuntimeError("test metrics appeared during LR selection")
    validation = [row for row in rows if row.get("val/loss_epoch")]
    state = torch.load(last_state, map_location="cpu", weights_only=False)
    observed_steps = [
        int(float(row["step"]))
        for row in rows
        if row.get("step") and row.get("train/loss_step")
    ]
    if not observed_steps or max(observed_steps) + 1 != a.steps:
        raise RuntimeError("terminal metrics step mismatch")
    checkpoint_state = next(
        value
        for key, value in state["callbacks"].items()
        if "'monitor': 'val/loss_epoch'" in str(key)
    )
    best_checkpoint = Path(checkpoint_state["best_model_path"]).resolve()
    best = min(validation, key=lambda row: float(row["val/loss_epoch"]))
    result = {
        "schema_version": 1,
        "status": "terminal_validation_cell",
        "model": a.model,
        "encoding": settings["encoding"],
        "fit_settings": settings,
        "learning_rate": a.learning_rate,
        "completed_optimizer_steps": a.steps,
        "selection_split": "validation",
        "test_set_accessed": False,
        "best_validation_step": int(float(best["step"])),
        "best_validation_loss": float(best["val/loss_epoch"]),
        "best_validation_r2": float(best["val/r2_all"]),
        "best_checkpoint": str(best_checkpoint.relative_to(versions[0].resolve())),
        "target_order": targets,
    }
    (a.output_dir / "cell.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
