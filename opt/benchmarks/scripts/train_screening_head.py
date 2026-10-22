"""Train one MIST sweep cell and publish validation-only evidence."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from .canonical_tasks import load, load_registry, resolve
from .checkpoint_selection import best_checkpoint, export_model
from .optimizer_options import add_optimizer_options


def training_precision():
    if not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def training_config(a, case):
    """Use the same model, optimizer, data, and training budget for every seed."""
    model = str(a.model.absolute())
    gradient_accumulation = a.effective_batch_size // a.batch_size
    return {
        "seed_everything": a.seed,
        "model": {
            "class_path": "electrolyte_fm.models.LMFinetuning",
            "init_args": {
                "output_size": len(case["targets"]),
                "encoder_ckpt": model,
                "tokenizer": model,
                "freeze_encoder": getattr(a, "freeze_encoder", False),
                "dropout": 0.2,
                "task": case["task"],
                "metrics": case["metrics"],
                "target_columns": case["targets"],
                "optimizer": {
                    "class_path": "torch.optim.AdamW",
                    "init_args": {
                        "lr": a.learning_rate,
                        "betas": list(getattr(a, "betas", (0.9, 0.999))),
                        "weight_decay": 0.01,
                    },
                },
                "lr_schedule": {
                    "class_path": "electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup",
                    "init_args": {
                        "num_training_steps": a.max_steps,
                        "num_warmup_steps": (
                            "beta2" if a.warmup_steps is None else a.warmup_steps
                        ),
                        "rel_decay": 0.1,
                    },
                },
            },
        },
        "data": {
            "class_path": ("opt.benchmarks.scripts.split_csv_data.SplitCSVDataModule"),
            "init_args": {
                "split_dir": str(a.dataset),
                "batch_size": a.batch_size,
                "val_batch_size": 64,
                "tokenizer": model,
                "target_columns": case["targets"],
                "encoding": getattr(a, "encoding", None) or case.get("encoding", "smiles-kekule"),
                "randomize": (
                    case.get("randomize", False)
                    if getattr(a, "randomize", None) is None else a.randomize
                ),
                "num_workers": a.num_workers,
                "prefetch_factor": 8 if a.num_workers else None,
            },
        },
        "trainer": {
            "max_steps": a.max_steps,
            "max_time": a.max_time,
            "val_check_interval": a.val_check_interval,
            "check_val_every_n_epoch": None,
            "precision": training_precision(),
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "strategy": "auto",
            "accumulate_grad_batches": gradient_accumulation,
            "enable_progress_bar": False,
            "default_root_dir": str(a.output_dir),
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                # Keep the checkpoint directory stable across scheduler retries.
                # New logger versions reset Lightning's best-checkpoint state.
                "init_args": {
                    "save_dir": str(a.output_dir),
                    "name": "metrics",
                    "version": 0,
                },
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--task-definitions", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--effective-batch-size", type=int, required=True)
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--encoding", choices=["smiles", "smiles-kekule"])
    p.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-steps", type=int, default=100000)
    add_optimizer_options(p, warmup_default="beta2")
    p.add_argument("--max-time", default="00:05:45:00")
    p.add_argument("--val-check-interval", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if (
        a.batch_size <= 0
        or a.effective_batch_size <= 0
        or a.effective_batch_size % a.batch_size
    ):
        p.error("effective batch size must be a positive multiple of batch size")
    gradient_accumulation = a.effective_batch_size // a.batch_size
    registry = load_registry(a.registry)
    raw_case = next(case for case in registry["cases"] if case["key"] == a.task)
    case = resolve(raw_case, load(a.task_definitions))
    a.targets = case["targets"]
    model = str(a.model.absolute())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = training_config(a, case)
    cfg = a.output_dir / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [
        sys.executable,
        "-m", "train",
        "fit",
        "--config",
        str(cfg),
    ]
    resumable = list(a.output_dir.rglob("*.ckpt"))
    if resumable:
        command.extend(
            ["--ckpt_path", str(max(resumable, key=lambda x: x.stat().st_mtime))]
        )
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])
    selected = best_checkpoint(a.output_dir)
    export_model(selected["checkpoint"], model, a.output_dir / "model",
                 encoding=config["data"]["init_args"]["encoding"])
    (a.output_dir / "cell.json").write_text(
        json.dumps(
            {
                "task": a.task,
                "seed": a.seed,
                "learning_rate": a.learning_rate,
                "betas": list(a.betas),
                "warmup_steps": a.warmup_steps,
                **selected,
                "microbatch_size": a.batch_size,
                "gradient_accumulation": gradient_accumulation,
                "effective_batch_size": a.effective_batch_size,
                "encoding": config["data"]["init_args"]["encoding"],
                "randomize": config["data"]["init_args"]["randomize"],
                "freeze_encoder": a.freeze_encoder,
                "max_steps": a.max_steps,
                "max_time": a.max_time,
                "val_check_interval": a.val_check_interval,
                "selection_split": "validation",
                "test_data_used": False,
                "model": model,
                "dataset": str(a.dataset),
                "targets": a.targets,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
