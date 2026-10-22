"""Fit a fresh conductivity head using the prepared AEM training protocol."""

import argparse
import json
from pathlib import Path

import torch

from electrolyte_fm.models.ionic_conductivity_model import IonicConductivityModel
from electrolyte_fm.models.prod_mixture import MISTIonicConductivity
from .checkpoint_selection import load_selected
from .training import run_fit, trainer_config


def training_config(args):
    """Express the existing AEM protocol for the shared training CLI."""
    base = str(args.model.resolve())
    return {
        "seed_everything": 42,
        "default_callbacks": False,
        "model": {
            "class_path": "electrolyte_fm.models.ionic_conductivity_model.IonicConductivityModel",
            "init_args": {
                "encoder_ckpt": base, "tokenizer": base,
                "n_components": 5, "freeze_encoder": False,
                "optimizer": {
                    "class_path": "torch.optim.AdamW",
                    "init_args": {"lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
                },
            },
        },
        "data": {
            "class_path": "electrolyte_fm.data_modules.mixture_dataset.ComponentDataModule",
            "init_args": {
                "path": str(args.dataset.resolve()), "tokenizer": base,
                "target_columns": ["ln k"], "n_components": 5,
                "include_temperature": True, "encoding": "smiles",
                "batch_size": 1, "val_batch_size": 4, "num_workers": args.num_workers,
            },
        },
        "trainer": trainer_config(args, optimizer_offload=True, gradient_checkpointing=True),
    }


def fit(args):
    use_gpu = torch.cuda.is_available()
    output = args.output_dir.resolve()
    base = str(args.model.resolve())
    fitted = run_fit(training_config(args), output)
    selected = load_selected(fitted["checkpoint"])
    # Training happens only in train.py; reconstruct on CPU for model export.
    module = IonicConductivityModel(
        encoder_ckpt=base, tokenizer=base, n_components=5, freeze_encoder=False,
    )
    module.cpu().float().load_state_dict(selected["state_dict"], strict=True)
    model = MISTIonicConductivity.from_components(
        module.encoder, module.task_network, tokenizer=module.tokenizer, n_components=5,
    )
    model.save_pretrained(output / "model")
    module.tokenizer.save_pretrained(output / "model")
    (output / "training.json").write_text(json.dumps({
        "dataset": str(args.dataset.resolve()), "base_model": base,
        "split_policy": "prepared-AEM-replacement-seed42", "seed": 42,
        "learning_rate": 3e-4, "batch_size": 1, "val_batch_size": 4,
        "freeze_encoder": False, "precision": args.precision,
        "gradient_checkpointing": True,
        "optimizer_cpu_offload": use_gpu,
        "max_steps": args.max_steps, "completed_steps": fitted["completed_steps"],
        "selected_step": selected["global_step"],
        "val_check_interval": args.val_check_interval,
        "selection_split": "validation", "test_data_used": False,
        "checkpoint": fitted["checkpoint"],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--precision", choices=["32-true", "bf16-mixed"], default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-time", default="00:07:30:00")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
