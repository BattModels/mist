#!/usr/bin/env python3
"""Evaluate a selected training run or released model on tmQM test once."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModel

from electrolyte_fm.data_modules.utils import MolEncoding
from electrolyte_fm.utils.tokenizer import load_tokenizer
from submit.submit import jsonnet_load


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    residual = pred - y
    denominator = float(np.square(y - y.mean()).sum())
    return {
        "n": int(len(y)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / denominator)
        if denominator
        else None,
    }


def select_validation_cell(paths: list[Path]) -> dict:
    cells = [json.loads(path.read_text()) for path in paths]
    if len(cells) < 2 or len({row["learning_rate"] for row in cells}) != len(cells):
        raise RuntimeError("expected at least two distinct learning-rate cells")
    models = {row["model"] for row in cells}
    if len(models) != 1:
        raise RuntimeError("learning-rate cells do not describe one model")
    for row in cells:
        if row.get("selection_split") != "validation":
            raise RuntimeError("cell selection split is not validation")
        if (
            row.get("test_set_accessed") is not False
            or row.get("completed_optimizer_steps") != 100000
        ):
            raise RuntimeError("ineligible validation cell")
    ranked = sorted(
        cells, key=lambda row: (row["best_validation_loss"], row["learning_rate"])
    )
    model = models.pop()
    return {
        "schema_version": 1,
        "status": "validation_selected",
        "model": model,
        "criterion": "minimum validation loss; test never read",
        "test_used_for_selection": False,
        "cells": cells,
        "selected": ranked[0],
    }


def select_validation_run(paths: list[Path], runs: list[Path]) -> tuple[dict, Path]:
    """Pair validation reports with their explicitly supplied training artifacts."""
    if len(paths) != len(runs):
        raise ValueError("--cells requires one --runs directory per cell")
    selection = select_validation_cell(paths)
    chosen = selection["selected"]
    run = runs[selection["cells"].index(chosen)].resolve()
    checkpoint = (run / chosen["best_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run):
        raise ValueError("selected checkpoint must be inside its training directory")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return selection, checkpoint


def export_model(model, tokenizer, output: Path, *, targets: list[str], encoding: str):
    """Save selected MIST weights and tokenizer as a standalone model directory."""
    from electrolyte_fm.models.prod_finetune import MISTFinetuned

    packaged = MISTFinetuned.from_components(
        model.encoder,
        model.task_network,
        model.transform,
        tokenizer=tokenizer,
        channels=targets,
        copy=False,
    )
    packaged.config.encoding = encoding
    packaged.save_pretrained(output)


def main() -> None:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--cells", nargs="+", type=Path)
    source.add_argument("--model")
    p.add_argument("--runs", nargs="+", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--export-model", action="store_true")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--expected-test-rows", type=int, default=10157)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    repository = Path(__file__).resolve().parents[3]
    submit_config = jsonnet_load(repository / "submit/tmQM.jsonnet")
    tmqm_config = submit_config["train"]
    targets = tmqm_config["data"]["init_args"]["target_columns"]
    if targets != tmqm_config["model"]["init_args"].get("target_columns", targets):
        raise RuntimeError("submit tmQM model/data target contracts disagree")
    selection_path = None
    if a.cells:
        # Avoid importing the training stack (and DeepSpeed's CUDA build probes)
        # when evaluating an already released Hugging Face checkpoint.
        from electrolyte_fm.models.lm_finetuning import LMFinetuning

        if not a.runs:
            p.error("--cells requires --runs")
        selection, checkpoint = select_validation_run(a.cells, a.runs)
        chosen = selection["selected"]
        batch_size = a.batch_size or int(chosen["fit_settings"]["batch_size"])
        max_sequence_length = int(chosen["fit_settings"]["max_sequence_length"])
        snapshot = a.model_dir
        encoding = chosen["encoding"]
        if checkpoint.is_file():
            model = LMFinetuning.load_from_checkpoint(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                strict=True,
                encoder_ckpt=str(snapshot),
                tokenizer=str(snapshot),
            )
        else:
            model = LMFinetuning.load(
                checkpoint, map_location=torch.device("cpu"), strict=True
            )
        selection_split = "validation"
        selection_path = a.output_dir / "selection.json"
    else:
        if a.export_model:
            p.error("--export-model requires --cells")
        snapshot, encoding = a.model_dir, "smiles"
        checkpoint = snapshot
        model = AutoModel.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        channels = [channel["name"] for channel in model.config.channels]
        if channels != targets:
            raise RuntimeError(f"released model channel mismatch: {channels}")
        batch_size = a.batch_size or 16
        max_sequence_length = None
        selection = {"model": a.model}
        selection_split = "released_checkpoint"
    stored = load_from_disk(str(a.dataset), keep_in_memory=False)
    if not isinstance(stored, DatasetDict):
        raise TypeError("tmQM release must be a saved DatasetDict")
    ds = stored["test"].to_iterable_dataset()
    source_rows, retained, encoded = 0, [], []
    converter = MolEncoding(encoding)
    for row in ds:
        source_rows += 1
        value = converter(str(row["smiles"]))
        if value is not None:
            retained.append(dict(row))
            encoded.append(value)
    if source_rows != a.expected_test_rows or not retained:
        raise RuntimeError("test split size or encoding coverage mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tokenizer = load_tokenizer(str(snapshot), local_files_only=True)
    if not a.cells:
        model.tokenizer = tokenizer
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            tokenizer_kwargs = {}
            if max_sequence_length is not None:
                tokenizer_kwargs["max_length"] = max_sequence_length
            batch = tokenizer(
                encoded[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch) if a.cells else model(**batch)
            predicted.extend(output.float().cpu().tolist())
    values = np.asarray(predicted, dtype=float)
    if values.shape != (len(retained), len(targets)) or not np.isfinite(values).all():
        raise RuntimeError("invalid held-out prediction matrix")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.export_model:
        export_model(
            model,
            tokenizer,
            a.output_dir / "model",
            targets=targets,
            encoding=encoding,
        )
    if selection_path is not None:
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    predictions = a.output_dir / "predictions.jsonl"
    with predictions.open("w") as stream:
        for row, prediction in zip(retained, values.tolist(), strict=True):
            stream.write(
                json.dumps(
                    {
                        "smiles": row["smiles"],
                        "target": {name: float(row[name]) for name in targets},
                        "prediction": dict(zip(targets, prediction, strict=True)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    metrics = {
        name: metric(
            np.asarray([float(row[name]) for row in retained]), values[:, index]
        )
        for index, name in enumerate(targets)
    }
    aggregate_r2 = float(np.mean([row["r2"] for row in metrics.values()]))
    result = {
        "schema_version": 1,
        "status": "sealed_test_complete",
        "model": selection["model"],
        "label": a.label,
        "selection": str(selection_path) if selection_path else None,
        "selection_split": selection_split,
        "test_used_for_selection": False,
        "test_access_count": 1,
        "source_rows": source_rows,
        "predicted_rows": len(retained),
        "coverage_fraction": len(retained) / source_rows,
        "target_order": targets,
        "targets": metrics,
        "mean_target_r2": aggregate_r2,
        "predictions": str(predictions),
        "checkpoint": str(checkpoint),
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
