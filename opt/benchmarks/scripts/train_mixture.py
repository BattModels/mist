"""Fit a released mixture split with the existing excess-property model."""

import argparse
import json
from pathlib import Path

from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsLightningModel,
    ExcessPhysicsModel,
    _default_targets,
)
from electrolyte_fm.utils.mixture_splits import split_files
from electrolyte_fm.utils.tokenizer import load_tokenizer
from .checkpoint_selection import load_selected
from .training import run_fit, trainer_config


def pretrained_mixture_model(name_or_path: str, architecture: dict) -> ExcessPhysicsModel:
    """Let the training CLI initialize the existing head from pretrained weights."""
    return ExcessPhysicsModel.from_pretrained_encoder(name_or_path, **architecture)


def training_config(args, architecture, thawing_stages):
    model = str(args.model.resolve())
    return {
        "seed_everything": 42,
        "default_callbacks": False,
        "model": {
            "class_path": "electrolyte_fm.models.excess_physics_model.ExcessPhysicsLightningModel",
            "init_args": {
                "model": {
                    "class_path": "opt.benchmarks.scripts.train_mixture.pretrained_mixture_model",
                    "init_args": {"name_or_path": model, "architecture": architecture},
                },
                "optimizer": {
                    "class_path": "torch.optim.AdamW",
                    "init_args": {"lr": args.learning_rate},
                },
                "lr_schedule": {
                    "class_path": "electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup",
                    "init_args": {
                        "num_training_steps": args.max_steps,
                        "num_warmup_steps": "beta2",
                    },
                },
            },
        },
        "data": {
            "class_path": "electrolyte_fm.data_modules.mixture_dataset.ComponentDataModuleFast",
            "init_args": {
                "path": str(args.dataset.resolve()),
                "tokenizer": model,
                "target_columns": _default_targets(),
                "batch_size": args.batch_size,
                "val_batch_size": getattr(args, "val_batch_size", 64),
                "encoding": "smiles-kekule",
                "randomize": True,
                "num_workers": args.num_workers,
            },
        },
        "trainer": trainer_config(args, monitor="val/loss", callbacks=[
            {
                "class_path": "electrolyte_fm.utils.progressive_thawing.ProgressiveThawing",
                "init_args": {"initial": ["model.encoder"], "stages": thawing_stages},
            },
            {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor"},
        ]),
    }


def fit(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    policy = getattr(args, "split_policy", "strict")
    splits = set(split_files(dataset))
    if policy == "strict":
        if not dataset.name.startswith("k-compound-strict-") or splits != {"train", "validation"}:
            raise ValueError(f"Expected a released strict train/validation fold: {dataset}")
    elif dataset.name != "random" or splits != {"train", "validation", "test"}:
        raise ValueError(f"Expected the released random configuration: {dataset}")

    # Reuse the released architecture, but initialize only from the pretrained
    # base: the random-split fine-tuned weights have seen strict validation rows.
    reference = json.loads((args.architecture / "config.json").read_text())
    architecture = {key: reference[key] for key in (
        "interactions", "num_control", "temperature_dependence",
        "relative_excess", "dropout",
    )}
    # Preserve the existing excess-model entrypoint's progressive thawing:
    # embeddings, then the last two encoder layers, one stage per epoch.
    layers = json.loads((args.model / "config.json").read_text())["num_hidden_layers"]
    thawing_stages = [["model.encoder.embeddings"]] + [
        [f"model.encoder.encoder.layer.{index}"]
        for index in range(layers - 1, max(-1, layers - 3), -1)
    ]
    selected_info = run_fit(training_config(args, architecture, thawing_stages), output)
    checkpoint = load_selected(selected_info["checkpoint"])
    selected = ExcessPhysicsLightningModel(model=checkpoint["hyper_parameters"]["model"])
    selected.load_state_dict(checkpoint["state_dict"])
    selected.model.save_pretrained(output / "model")
    load_tokenizer(str(args.model.resolve())).save_pretrained(output / "model")
    (output / "training.json").write_text(json.dumps({
        "dataset": str(dataset), "fold": dataset.name,
        "base_model": str(args.model.resolve()),
        "architecture_model": str(args.architecture.resolve()),
        "architecture": architecture, "seed": 42,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * getattr(args, "accumulate_grad_batches", 1),
        "split_policy": policy,
        "max_steps": args.max_steps, "completed_steps": selected_info["completed_steps"],
        "val_check_interval": args.val_check_interval,
        "validation_loss": selected_info["validation_loss"],
        "checkpoint": selected_info["checkpoint"],
        "selection_split": "validation", "test_data_used": False,
        "normalization_split": "train",
        "thawing_stages": thawing_stages,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=["32-true", "bf16-mixed"], default="32-true")
    parser.add_argument("--split-policy", choices=["strict", "random"], default="strict")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--val-check-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-time", default="00:07:30:00")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
