"""Configuration and checkpoint plumbing for the shared train.py entrypoint."""

import math
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DeepSpeedStrategy

from .checkpoint_selection import checkpoint_metadata


class EncoderGradientCheckpointing(Callback):
    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            pl_module.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


def trainer_config(args, *, monitor="val/loss_epoch", callbacks=(),
                   optimizer_offload=False, gradient_checkpointing=False):
    output = args.output_dir.resolve()
    use_gpu = torch.cuda.is_available()
    configured_callbacks = [{
        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
        "init_args": {
            "dirpath": str(output / "checkpoints"), "filename": "step={step}",
            "monitor": monitor, "mode": "min", "save_top_k": 1, "save_last": True,
            "auto_insert_metric_name": False,
        },
    }, *callbacks, {
        "class_path": "electrolyte_fm.utils.callbacks.job_timer.JobTimer",
        "init_args": {"duration": args.max_time},
    }]
    if gradient_checkpointing:
        configured_callbacks.append({
            "class_path": "opt.benchmarks.scripts.training.EncoderGradientCheckpointing",
        })
    strategy = "auto"
    if optimizer_offload and use_gpu:
        offload = DeepSpeedStrategy(stage=2, offload_optimizer=True, pin_memory=True)
        offload.config["zero_force_ds_cpu_optimizer"] = False
        strategy = {
            "class_path": "lightning.pytorch.strategies.DeepSpeedStrategy",
            "init_args": {"config": offload.config},
        }
    return {
        "accelerator": "gpu" if use_gpu else "cpu", "devices": 1,
        "strategy": strategy, "precision": getattr(args, "precision", "32-true"),
        "accumulate_grad_batches": getattr(args, "accumulate_grad_batches", 1),
        "max_steps": args.max_steps, "max_epochs": -1,
        "val_check_interval": args.val_check_interval,
        "check_val_every_n_epoch": None, "enable_progress_bar": False,
        "callbacks": configured_callbacks,
        "logger": {
            "class_path": "lightning.pytorch.loggers.CSVLogger",
            "init_args": {"save_dir": str(output), "name": "metrics", "version": 0},
        },
        "default_root_dir": str(output),
    }


def run_fit(config, output):
    """Run train.py, then require completion and select by saved validation state."""
    # These wrappers fully specify construction. Some existing model classes
    # save resolved Python objects in hparams that cannot round-trip through CLI
    # argument parsing; Lightning still restores their complete training state.
    config = dict(config, resume_from_config=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-m", "train",
               "fit", "--config", str(config_path), "--weights_only=false"]
    last = output / "checkpoints/last.ckpt"
    if last.exists():
        command.extend(["--ckpt_path", str(last)])
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])
    state = checkpoint_metadata(last)
    completed_steps = state["global_step"]
    if completed_steps < config["trainer"]["max_steps"]:
        raise RuntimeError("Time limit reached; resume the saved last.ckpt to finish fitting")
    monitor = next(callback["init_args"]["monitor"]
                   for callback in config["trainer"]["callbacks"]
                   if callback["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint")
    candidates = []
    for callback in state.get("callbacks", {}).values():
        if callback.get("monitor") != monitor:
            continue
        for filename, value in callback.get("best_k_models", {}).items():
            path, score = Path(filename), float(value)
            if path.exists() and math.isfinite(score):
                candidates.append((score, str(path)))
    if not candidates:
        raise RuntimeError("No checkpoint with finite validation loss")
    score, checkpoint = min(candidates)
    return {"checkpoint": checkpoint, "validation_loss": score,
            "completed_steps": completed_steps}
