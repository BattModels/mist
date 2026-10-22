"""Bounded PubChemQC charge-head fitting, validation selection, and inference."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rdkit import Chem

from electrolyte_fm.data_modules.pubchem_qc import (
    DEFAULT_TOKEN_TARGETS,
    PubChemQC,
    mol_from_prediction,
    serialize_molecule,
)
from electrolyte_fm.models.lm_finetuning import load_encoder
from electrolyte_fm.models.prediction_task_head import PredictionTaskHead, TokenTaskHead
from electrolyte_fm.models.prod_finetune import build_encoder_from_dict, encoder_config
from electrolyte_fm.models.token_level import TokenLevelPredictor
from electrolyte_fm.utils.lr_schedule import RelativeCosineWarmup
from electrolyte_fm.utils.stage_hf import link_model
from electrolyte_fm.utils.tokenizer import load_tokenizer

from .optimizer_options import add_optimizer_options


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def build_model(
    snapshot, learning_rate=2e-5, *, betas=(0.9, 0.999),
    warmup_steps=None, max_steps=100000,
):
    encoder = (
        snapshot
        if isinstance(snapshot, torch.nn.Module)
        else load_encoder(snapshot, tokenizer=snapshot)
    )
    size = encoder.config.hidden_size
    return TokenLevelPredictor(
        encoder=encoder,
        seq_network=PredictionTaskHead(size, 1),
        seq_targets=["charge"],
        token_network=TokenTaskHead(size, 2 * len(DEFAULT_TOKEN_TARGETS)),
        optimizer=partial(
            torch.optim.AdamW, lr=learning_rate, betas=tuple(betas), weight_decay=0.01,
        ),
        lr_schedule=(
            partial(
                RelativeCosineWarmup, num_training_steps=max_steps,
                num_warmup_steps=warmup_steps, rel_decay=0.1,
            ) if warmup_steps is not None else None
        ),
        freeze_encoder=False,
        # The paper's charge experiment optimizes masked token MSE only, in e².
        # Keep the existing model's optional sequence branch out of this loss.
        sequence_loss_weight=0.0,
    )


def save_model(model, tokenizer, output):
    """Save charge weights, architecture, and tokenizer in one inference directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"encoder": encoder_config(model.encoder)})
    torch.save(model.state_dict(), output / "model.pt")
    tokenizer.save_pretrained(output)


def restore(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = build_model(build_encoder_from_dict(config["encoder"]))
    state = torch.load(
        directory / "model.pt", map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(state, strict=True)
    return model.eval(), load_tokenizer(str(directory), local_files_only=True)


def data_module(dataset, snapshot, batch_size, workers=4):
    if not Path(dataset).is_dir():
        raise FileNotFoundError(f"Missing staged PubChemQC JCIM2017 corpus directory: {dataset}")
    return PubChemQC(
        path=str(dataset),
        tokenizer=snapshot,
        batch_size=batch_size,
        num_workers=workers,
        randomize=True,
        evaluation_randomize=False,
        smear_hydrogens=True,
        seq_targets=["charge"],
        include_3d=False,
    )


def fit(args):
    pl.seed_everything(42, workers=True)
    snapshot = str(args.model.absolute())
    betas = getattr(args, "betas", (0.9, 0.999))
    warmup_steps = getattr(args, "warmup_steps", None)
    model = build_model(
        snapshot, args.learning_rate, betas=betas,
        warmup_steps=warmup_steps, max_steps=args.max_steps,
    )
    microbatch = args.batch_size or (
        4 if model.encoder.config.hidden_size > 1024 else 32
    )
    data = data_module(args.dataset, snapshot, microbatch, args.num_workers)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    best = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
        auto_insert_metric_name=False,
    )
    # At most one best and one last checkpoint, including optimizer state for
    # restart. Their validation score is read from callback state, never a name.
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "32-true",
        default_root_dir=output,
        logger=CSVLogger(output, name="metrics"),
        callbacks=[best],
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_epochs=1000,
        accumulate_grad_batches=args.gradient_accumulation,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    last = output / "checkpoints" / "last.ckpt"
    # Lightning 2.6 exposes this explicitly; earlier supported versions load
    # their own pickled hyperparameters without the weights-only restriction.
    load_options = (
        {"weights_only": False}
        if "weights_only" in inspect.signature(trainer.fit).parameters
        else {}
    )
    trainer.fit(
        model.train(),
        data,
        ckpt_path=str(last) if last.is_file() else None,
        **load_options,
    )
    if not best.best_model_path or best.best_model_score is None:
        raise RuntimeError("No validation checkpoint saved within the run budget")
    state = torch.load(
        best.best_model_path, map_location="cpu", weights_only=False, mmap=True
    )
    model.cpu().load_state_dict(state["state_dict"], strict=True)
    save_model(model, load_tokenizer(snapshot, local_files_only=True), output / "model")
    write_json(
        output / "cell.json",
        {
            "task": args.task,
            "model": snapshot,
            "learning_rate": args.learning_rate,
            "betas": list(betas),
            "warmup_steps": warmup_steps,
            "checkpoint": best.best_model_path,
            "validation_loss": float(best.best_model_score),
            "selection_split": "validation",
            "test_data_used": False,
            "freeze_encoder": False,
            "targets": model.token_targets,
            "microbatch_size": microbatch,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": microbatch * args.gradient_accumulation,
            "global_step": trainer.global_step,
        },
    )


def select_cell(paths, models):
    if len(paths) != len(models):
        raise ValueError("one model directory is required per LR cell")
    candidates = [
        (json.loads(Path(path).read_text()), model)
        for path, model in zip(paths, models, strict=True)
    ]
    if not candidates:
        raise ValueError("No LR cells to select")
    for cell, _ in candidates:
        if (
            cell.get("selection_split") != "validation"
            or cell.get("test_data_used") is not False
        ):
            raise ValueError("Charge-head selection must use validation only")
        if not math.isfinite(cell["validation_loss"]):
            raise ValueError("Non-finite validation loss")
    if len({cell["task"] for cell, _ in candidates}) != 1:
        raise ValueError("Cannot select across different models/tasks")
    return min(
        candidates,
        key=lambda candidate: (
            candidate[0]["validation_loss"], candidate[0]["learning_rate"]
        ),
    )


@torch.inference_mode()
def score_batches(model, batches, device):
    count = torch.zeros(4, dtype=torch.float64)
    absolute = torch.zeros_like(count)
    square = torch.zeros_like(count)
    rows = 0
    for batch in batches:
        _, prediction = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        target = batch["token_target"].double()
        mask = batch["token_target_mask"].bool()
        error = (prediction.cpu().double() - target).masked_fill(~mask, 0)
        absolute += error.abs().sum((0, 1))
        square += error.square().sum((0, 1))
        count += mask.sum((0, 1))
        rows += len(target)
    if not (count > 0).all():
        raise ValueError("Test set has no labels for one or more charge channels")
    return (
        {
            "mae": float(absolute.sum() / count.sum()),
            "rmse": float((square.sum() / count.sum()).sqrt()),
        },
        {
            name: {"mae": float(a / n), "rmse": float((s / n).sqrt()), "count": int(n)}
            for name, a, s, n in zip(
                model.token_targets, absolute, square, count, strict=True
            )
        },
        rows,
    )


def evaluate(args):
    selected, selected_model = select_cell(args.cells, args.models)
    model_dir = args.output_dir / "model"
    link_model(selected_model, model_dir, link_files=True)
    selection = args.output_dir / "selection.json"
    write_json(selection, {"selected": selected, "selection_test_used": False})
    model, _ = restore(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = args.batch_size or (
        8 if model.encoder.config.hidden_size > 1024 else 64
    )
    data = data_module(
        args.dataset, str(model_dir), batch_size, args.num_workers
    )
    data.setup("test")
    metrics, channels, rows = score_batches(model, data.test_dataloader(), device)
    write_json(
        args.output_dir / "metrics.json",
        {
            "case": args.task,
            "base": args.base,
            "dataset": "pubchemqc",
            "split": "test",
            "selected_by": "validation only",
            "provenance": "PubChemQC JCIM2017; validation-selected unfrozen LR sweep",
            "metrics": metrics,
            "channels": channels,
            "rows": rows,
            "unit": "elementary charge",
            "selection": selected,
        },
    )


@torch.inference_mode()
def predict_rows(model, tokenizer, smiles, batch_size):
    device = next(model.parameters()).device
    for start in range(0, len(smiles), batch_size):
        batch = tokenizer(
            smiles[start : start + batch_size], padding=True, return_tensors="pt"
        ).to(device)
        _, tokens = model(batch["input_ids"], batch["attention_mask"])
        for ids, charges in zip(batch["input_ids"].cpu(), tokens.cpu(), strict=True):
            molecule = mol_from_prediction(
                ids, tokenizer, y_token=charges, token_targets=DEFAULT_TOKEN_TARGETS
            )
            row = serialize_molecule(molecule)
            if not row["atoms"] or any(
                target not in atom["properties"]
                for atom in row["atoms"]
                for target in DEFAULT_TOKEN_TARGETS
            ):
                raise ValueError("Prediction is missing atom-level charge annotations")
            yield row


def predict(args):
    model, tokenizer = restore(args.model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    with args.molecules.open() as stream:
        smiles = [row["smi"] for row in csv.DictReader(stream)]
    source_rows = len(smiles)
    # The design table includes aliases for the same molecule. Predict once per
    # structure so its downstream join cannot multiply those rows.
    smiles = list(
        dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles)
    )
    size = args.batch_size or (8 if model.encoder.config.hidden_size > 1024 else 64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "electrolytes.jsonl"
    with output.open("w") as stream:
        for row in predict_rows(model, tokenizer, smiles, size):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(
        args.output_dir / "receipt.json",
        {
            "molecules": str(args.molecules),
            "rows": len(smiles),
            "source_rows": source_rows,
            "model": str(args.model),
            "predictions": str(output),
            "source_dataset": "PubChemQC JCIM2017",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    fitting = modes.add_parser("fit")
    fitting.add_argument("--model", type=Path, required=True)
    fitting.add_argument("--learning-rate", type=float, required=True)
    # Existing charge sweeps used a constant LR. Only the new companion arrays
    # opt into beta2 warmup and cosine decay.
    add_optimizer_options(fitting, warmup_default=None)
    fitting.add_argument("--gradient-accumulation", type=int, default=8)
    fitting.add_argument("--max-steps", type=int, default=100000)
    fitting.add_argument("--max-time", default="00:05:45:00")
    fitting.add_argument("--val-check-interval", type=int, default=1000)
    testing = modes.add_parser("evaluate")
    testing.add_argument("--cells", nargs="+", type=Path, required=True)
    testing.add_argument("--models", nargs="+", type=Path, required=True)
    testing.add_argument("--base", required=True)
    for command in (fitting, testing):
        command.add_argument("--task", required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--num-workers", type=int, default=4)
    prediction = modes.add_parser("predict")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--molecules", type=Path, required=True)
    for command in (fitting, testing, prediction):
        command.add_argument("--batch-size", type=int)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"fit": fit, "evaluate": evaluate, "predict": predict}[args.mode](args)


if __name__ == "__main__":
    main()
