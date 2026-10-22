"""Score one selected model once on its held-out test split."""

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from electrolyte_fm.data_modules.utils import MolEncoding
from electrolyte_fm.models.prod_finetune import load_finetuned
from electrolyte_fm.utils.stage_hf import link_model

from .canonical_tasks import load, load_registry, resolve
from .checkpoint_selection import select_lr


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--cells", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task-definitions", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    raw = next(case for case in registry["cases"] if case["key"] == args.case)
    case = resolve(raw, load(args.task_definitions))
    if args.cells:
        if not args.models or len(args.cells) != len(args.models):
            parser.error("--cells requires one --models directory per cell")
        candidates = [
            (json.loads(path.read_text()), model)
            for path, model in zip(args.cells, args.models, strict=True)
        ]
        chosen, model_dir = candidates[select_lr([item for item, _ in candidates])]
        candidates.sort(
            key=lambda item: (item[0]["validation_loss"], item[0]["learning_rate"])
        )
        selection = {
            "provenance": "new_lr_sweep",
            "test_used": False,
            "selected": chosen,
            "ranked_validation": [item for item, _ in candidates],
        }
    else:
        release = case["release"]
        if not release:
            parser.error("--model requires a released model in the registry")
        model_dir = args.model
        selection = {
            "provenance": "released",
            "test_used": False,
            "selected": {"validation_loss": release["validation_value"]},
            "release": {"id": release["id"], "revision": release["revision"]},
        }
    # Persist the validation-only choice before opening held-out measurements.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    link_model(model_dir, args.output_dir / "model", link_files=True)
    model = load_finetuned(str(args.output_dir / "model"), local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)
    frame = pd.read_csv(args.test_csv)
    input_rows = len(frame)
    # Match the representation/invalid-SMILES filtering used by the fit script.
    encoding = MolEncoding(model.encoding)
    smiles = frame.smiles.map(
        lambda smi: encoding(smi) if isinstance(smi, str) else None
    )
    valid = smiles.notna()
    frame = frame.loc[valid]
    if frame.empty:
        raise ValueError("No test molecules can be encoded")
    predictions = model.predict(smiles.loc[valid].tolist())
    # Align by name even when a release has the right number of channels in a
    # different order; width alone cannot establish the target correspondence.
    pred = np.column_stack([predictions[name]["value"] for name in case["targets"]])
    if not np.isfinite(pred).all():
        raise ValueError("The selected model produced non-finite test predictions")
    target = frame[case["targets"]].to_numpy(float)
    if case["task"] == "binary":
        scores = {"auroc": [], "avg-precision": []}
        for index in range(target.shape[1]):
            mask = np.isfinite(target[:, index]) & np.isfinite(pred[:, index])
            if len(np.unique(target[mask, index])) > 1:
                scores["auroc"].append(
                    roc_auc_score(target[mask, index], pred[mask, index])
                )
                scores["avg-precision"].append(
                    average_precision_score(target[mask, index], pred[mask, index])
                )
        metrics = {
            key: float(np.mean(values)) if values else None
            for key, values in scores.items()
        }
    else:
        mask = np.isfinite(target)
        metrics = {
            "mae": float(mean_absolute_error(target[mask], pred[mask])),
            "rmse": float(mean_squared_error(target[mask], pred[mask]) ** 0.5),
        }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "base": raw["base"],
                "dataset": case["dataset"],
                "variant": case.get("variant"),
                "split": "test",
                "selected_by": "validation only",
                "provenance": selection.get("provenance", "released"),
                "selected": selection["selected"],
                "metrics": metrics,
                "input_rows": input_rows,
                "predicted_rows": len(frame),
                "encoding": encoding.value,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
