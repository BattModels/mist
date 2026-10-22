"""Evaluate a seed-42 LR winner and fit additional seeds without checkpoints."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.cli import LightningCLI

from electrolyte_fm.models import LMFinetuning
from electrolyte_fm.models.prod_finetune import load_finetuned

from .canonical_tasks import load, load_registry, resolve
from .checkpoint_selection import select_lr
from .split_csv_data import SplitCSVDataModule
from .train_screening_head import training_config, training_precision


class BestValidationWeights(Callback):
    """Retain one validation-selected state in CPU memory, never on disk."""

    def __init__(self):
        self.loss = math.inf
        self.step = None
        self.weights = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        loss = float(trainer.callback_metrics["val/loss_epoch"])
        if math.isfinite(loss) and loss < self.loss:
            self.loss, self.step = loss, trainer.global_step
            self.weights = {
                key: value.detach().to("cpu", copy=True)
                for key, value in pl_module.state_dict().items()
            }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate_losses(trainer, module, data_args, test_csv):
    # This datamodule is first constructed after validation selects the weights.
    data = SplitCSVDataModule(**data_args, test_csv=str(test_csv))
    validation = trainer.validate(module, datamodule=data, verbose=False)[0]
    test = trainer.test(module, datamodule=data, verbose=False)[0]
    losses = {
        "validation_loss": validation["val/loss_epoch"],
        "test_loss": test["test/loss_epoch"],
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise ValueError("Validation and test losses must be finite")
    return losses


def selected_training_settings(cell, case):
    """Recover the complete recipe, including defaults of the original cells."""
    microbatch = case.get("microbatch", 16 if case["base"] == "mist-28m" else 1)
    return {
        "learning_rate": cell["learning_rate"],
        "betas": cell.get("betas", [0.9, 0.999]),
        "warmup_steps": cell.get("warmup_steps", "beta2"),
        "freeze_encoder": cell.get("freeze_encoder", False),
        "encoding": cell.get("encoding", case.get("encoding", "smiles-kekule")),
        "randomize": cell.get("randomize", case.get("randomize", False)),
        "batch_size": int(cell.get("microbatch_size", microbatch)),
        "effective_batch_size": int(
            cell.get("effective_batch_size", case.get("effective_batch", 128))
        ),
        "max_steps": cell.get("max_steps", 100000),
        "max_time": cell.get("max_time", "00:05:45:00"),
        "val_check_interval": cell.get("val_check_interval", 1000),
    }


def seed42(args, case):
    cells = [json.loads(path.read_text()) for path in args.cells]
    if len(cells) != len(args.models):
        raise ValueError("Expected one model artifact per LR cell")
    if any(cell.get("seed", 42) != 42 or cell["task"] != args.task for cell in cells):
        raise ValueError("Expected seed-42 LR cells for this task")
    index = select_lr(cells)
    chosen = cells[index]
    selection = {
        "task": args.task,
        "seed": 42,
        "learning_rate": chosen["learning_rate"],
        "validation_loss": chosen["validation_loss"],
        "selection_split": "validation",
        "test_data_used": False,
        "training": selected_training_settings(chosen, case),
    }
    write_json(args.output_dir / "selection.json", selection)
    # Model paths remain explicit artifact arguments rather than receipt fields.
    path = args.models[index]
    model = load_finetuned(str(path), local_files_only=True)
    module = LMFinetuning(
        output_size=len(case["targets"]),
        encoder_ckpt=str(path),
        tokenizer=str(path),
        task=case["task"],
        metrics=case["metrics"],
        target_columns=case["targets"],
    )
    module.encoder, module.task_network, module.transform = (
        model.encoder,
        model.task_network,
        model.transform,
    )
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=training_precision(),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    data_args = {
        "split_dir": str(args.dataset),
        "tokenizer": str(path),
        "target_columns": case["targets"],
        "val_batch_size": 64,
        "encoding": model.encoding,
        "num_workers": args.num_workers,
    }
    losses = evaluate_losses(trainer, module, data_args, args.test_csv)
    write_json(
        args.output_dir / "losses.json",
        {
            **selection,
            **losses,
            "base": case["base"],
            "dataset": case["dataset"],
            "selection_validation_loss": chosen["validation_loss"],
        },
    )


def fit(args, case):
    selection = json.loads(args.selection.read_text())
    if (
        selection["task"] != args.task
        or selection["seed"] != 42
        or selection.get("selection_split") != "validation"
        or selection.get("test_data_used") is not False
    ):
        raise ValueError("Expected a validation-only LR selection from seed 42")
    args.learning_rate = selection["learning_rate"]
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Selected learning rate must be finite and positive")
    # The four repeats must inherit the winning beta/freeze/augmentation recipe,
    # not merely its LR. Legacy selections without a recipe keep CLI defaults.
    for key, value in selection.get("training", {}).items():
        setattr(args, key, value)
    config = training_config(args, case)
    config["trainer"]["enable_checkpointing"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config)
    # Reuse the same CLI model/data construction, with checkpointing disabled.
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(
        args=config,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
    best = BestValidationWeights()
    cli.trainer.callbacks.append(best)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    if best.weights is None:
        raise RuntimeError("No finite validation loss was recorded during fitting")
    cli.model.load_state_dict(best.weights)
    best.weights = None
    losses = evaluate_losses(
        cli.trainer, cli.model, config["data"]["init_args"], args.test_csv
    )
    write_json(
        args.output_dir / "losses.json",
        {
            "task": args.task,
            "base": case["base"],
            "dataset": case["dataset"],
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "training": selection.get("training"),
            "lr_selection_seed": 42,
            "selection_split": "validation",
            "test_data_used": False,
            "selection_validation_loss": best.loss,
            "selected_step": best.step,
            "completed_steps": cli.trainer.global_step,
            **losses,
        },
    )


def summarize(paths):
    rows = sorted(
        (json.loads(path.read_text()) for path in paths), key=lambda row: row["seed"]
    )
    if [row["seed"] for row in rows] != [42, 43, 44, 45, 46]:
        raise ValueError("Expected exactly one result for each seed 42–46")
    for key in ("task", "base", "dataset", "learning_rate", "training"):
        if len({json.dumps(row.get(key), sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"Seed results disagree on {key}")
    summary = {}
    for key in ("validation_loss", "test_loss"):
        values = [row[key] for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {key}")
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "sem": statistics.stdev(values) / math.sqrt(len(values)),
        }
    return {
        key: rows[0][key] for key in ("task", "base", "dataset", "learning_rate")
    } | {
        "n_seeds": len(rows),
        "seeds": rows,
        "summary": summary,
        "uncertainty": "sample standard deviation and standard error across training seeds; fixed data splits",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("seed42")
    select.add_argument("--cells", type=Path, nargs="+", required=True)
    select.add_argument("--models", type=Path, nargs="+", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--selection", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=[43, 44, 45, 46], required=True)
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument("--effective-batch-size", type=int, required=True)
    train.add_argument("--max-steps", type=int, default=100000)
    train.add_argument("--warmup-steps", type=int)
    train.add_argument("--max-time", default="00:05:45:00")
    train.add_argument("--val-check-interval", type=int, default=1000)
    for command in (select, train):
        command.add_argument("--task", required=True)
        command.add_argument("--registry", type=Path, required=True)
        command.add_argument("--task-definitions", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--test-csv", type=Path, required=True)
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--output-dir", type=Path, required=True)
    collate = commands.add_parser("summarize")
    collate.add_argument("--losses", type=Path, nargs="+", required=True)
    collate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        write_json(args.output, summarize(args.losses))
        return
    registry = load_registry(args.registry)
    raw = next(case for case in registry["cases"] if case["key"] == args.task)
    case = resolve(raw, load(args.task_definitions))
    if args.command == "seed42":
        seed_everything(42, workers=True)
        seed42(args, case)
    else:
        if (
            args.batch_size <= 0
            or args.effective_batch_size <= 0
            or args.effective_batch_size % args.batch_size
        ):
            parser.error(
                "effective batch size must be a positive multiple of batch size"
            )
        fit(args, case)


if __name__ == "__main__":
    main()
