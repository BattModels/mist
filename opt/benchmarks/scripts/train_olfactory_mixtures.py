"""Fit the existing olfactory mixture architecture on the public seed-42 split."""

import argparse
import json
from pathlib import Path

import torch
from .checkpoint_selection import load_selected
from .training import run_fit, trainer_config


def training_config(args):
    base = str(args.model.resolve())
    # Keep the existing WeightedSumPooling + ScaledCosineDistance architecture.
    # Distances arrive in [0, 1]; identity preserves these raw targets for the
    # existing loss, which calls the normalizer's inverse on batch targets.
    return {
        "seed_everything": 42,
        "default_callbacks": False,
        "model": {
            "class_path": "electrolyte_fm.models.MixtureSimilarityModel",
            "init_args": {
                "encoder_ckpt": base, "distance_metric": "scaled_cosine",
                "freeze_encoder": False, "target_columns": ["value"],
                "transform": "identity", "metrics": ["mae"],
                "optimizer": {
                    "class_path": "torch.optim.AdamW",
                    "init_args": {"lr": args.learning_rate, "weight_decay": 0.01},
                },
                "lr_schedule": {
                    "class_path": "electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup",
                    "init_args": {
                        "num_training_steps": args.max_steps,
                        "num_warmup_steps": "beta2", "rel_decay": 0.1,
                    },
                },
            },
        },
        "data": {
            "class_path": "electrolyte_fm.data_modules.MultiMixtureDataModule",
            "init_args": {
                "path": str(args.dataset.resolve()), "tokenizer": base,
                "target_columns": ["value"], "encoding": "smiles-kekule",
                "batch_size": args.batch_size, "val_batch_size": args.val_batch_size,
                "num_workers": args.num_workers,
            },
        },
        "trainer": trainer_config(
            args, optimizer_offload=args.optimizer_offload,
            gradient_checkpointing=args.optimizer_offload and torch.cuda.is_available(),
        ),
    }


def fit(args):
    if args.batch_size < 2:
        raise ValueError("The existing R2 loss requires at least two observations per batch")
    output = args.output_dir.resolve()
    base = args.model.resolve()
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"A staged local encoder is required: {base}")
    fitted = run_fit(training_config(args), output)
    selected = load_selected(fitted["checkpoint"])
    # Keep resumption state in checkpoints/; publish just the selected inference
    # checkpoint, with the hyperparameters and normalization needed to load it.
    for key in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
        selected.pop(key, None)
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected, model_dir / "model.ckpt")
    (output / "training.json").write_text(json.dumps({
        "dataset": str(args.dataset.resolve()), "base_model": str(base),
        "seed": 42, "split_policy": "random-observation-80-10-10",
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size, "precision": args.precision,
        "transform": "identity", "freeze_encoder": False,
        "optimizer_cpu_offload": args.optimizer_offload and torch.cuda.is_available(),
        "max_steps": args.max_steps, "completed_steps": fitted["completed_steps"],
        "selected_step": selected["global_step"], "selection_split": "validation",
        "test_evaluated": False,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    # Baseline optimizer and batch sizes come from submit/chemixhub.jsonnet.
    parser.add_argument("--learning-rate", type=float, default=1.6e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--precision", choices=["32-true", "bf16-mixed"], default="32-true")
    parser.add_argument("--optimizer-offload", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-time", default="00:07:30:00")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
