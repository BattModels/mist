"""Extract local model artifacts and analyze prepared embedding cohorts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def token_weights(
    model_dir: Path, output_dir: Path, *, fallback_tokenizer: Path | None = None
) -> None:
    """Read just the vocabulary tensor, including sharded model releases."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file() and fallback_tokenizer is not None:
        tokenizer = fallback_tokenizer
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Missing tokenizer artifact: {tokenizer}")
    candidates = []
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        candidates = [
            (model_dir / file, key)
            for key, file in json.loads(index.read_text())["weight_map"].items()
            if key.endswith(".word_embeddings.weight")
        ]
    else:
        weights = model_dir / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing model weights at {model_dir}")
        with safe_open(weights, framework="pt", device="cpu") as source:
            candidates = [
                (weights, key)
                for key in source.keys()  # noqa: SIM118 - safe_open is not a dict.
                if key.endswith(".word_embeddings.weight")
            ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one word-embedding tensor, found {len(candidates)}")
    file, key = candidates[0]
    with safe_open(file, framework="pt", device="cpu") as source:
        embeddings = source.get_tensor(key).float().contiguous()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file({key: embeddings}, output_dir / "model.safetensors")
    shutil.copyfile(tokenizer, output_dir / "tokenizer.json")


def read_cohort(path: Path, panel: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared {panel} cohort: {path}")
    frame = pd.read_csv(path)
    labels = (
        ["Aromaticity", "Number of Benzene"]
        if panel == "aromaticity"
        else ["Condensed"]
    )
    columns = ["molecule_id", "smiles", "split", *labels]
    if (
        not set(columns) <= set(frame)
        or frame.empty
        or frame[columns].isna().any().any()
    ):
        raise ValueError(f"{path} requires nonempty {columns}")
    if frame.molecule_id.duplicated().any():
        raise ValueError("Cohort molecule IDs must be unique")
    if set(frame.split) != {"train", "validation"}:
        raise ValueError(
            "Provide train/validation membership in the split column"
        )
    label = labels[0]
    if frame[label].nunique() != 2 or any(
        group[label].nunique() != 2 for _, group in frame.groupby("split")
    ):
        raise ValueError("Each split must contain both chemical classes")
    return frame


def save_embeddings(output: Path, embeddings: np.ndarray, molecule_ids) -> None:
    """Store identifiers as Unicode so the artifact never requires pickle."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=embeddings,
        molecule_id=np.asarray([str(value) for value in molecule_ids], dtype=str),
    )


def encode(args) -> None:
    import torch
    from transformers import AutoModel

    from electrolyte_fm.data_modules.utils import MolEncoding
    from electrolyte_fm.models.prod_finetune import encode_inputs
    from electrolyte_fm.utils.tokenizer import load_tokenizer

    frame = read_cohort(args.cohort, args.panel)
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"Supply a staged local MIST encoder at {args.model}")
    smiles = [MolEncoding("smiles-kekule")(value) for value in frame.smiles]
    if any(value is None for value in smiles):
        raise ValueError("Cohort contains invalid SMILES; no rows are dropped")
    tokenizer = load_tokenizer(str(args.model), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModel.from_pretrained(
            str(args.model),
            local_files_only=True,
            trust_remote_code=True,
            add_pooling_layer=False,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        .eval()
        .to(device)
    )
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            inputs = tokenizer(
                smiles[start : start + args.batch_size],
                padding=True,
                return_tensors="pt",
            ).to(device)
            embeddings.append(
                encode_inputs(model, inputs["input_ids"], inputs["attention_mask"])[
                    :, 0, :
                ]
                .float()
                .cpu()
                .numpy()
            )
    values = np.concatenate(embeddings)
    if (
        values.shape != (len(frame), model.config.hidden_size)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid embedding dimensions or nonfinite values")
    save_embeddings(args.output, values, frame.molecule_id)


def analyze(args) -> None:
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from opt.interp_embeddings.shallow_models import (
        evaluate_on_test,
        linear_separability_diagnostics,
    )

    frame = read_cohort(args.cohort, args.panel)
    with np.load(args.embeddings, allow_pickle=False) as stored:
        if stored["molecule_id"].tolist() != frame.molecule_id.astype(str).tolist():
            raise ValueError("Embedding rows must match the exact cohort order")
        embeddings = stored["embeddings"]
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(frame)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid cohort embedding matrix")
    if not args.settings.is_file():
        raise FileNotFoundError(f"Missing UMAP/SVM settings: {args.settings}")
    settings = json.loads(args.settings.read_text())
    projection = settings["umap"]
    if not {
        "n_neighbors",
        "min_dist",
        "metric",
        "random_state",
        "transform_seed",
    } <= set(projection):
        raise ValueError("Provide explicit UMAP parameters and seeds")
    if projection.get("n_components", 2) != 2:
        raise ValueError("Paper UMAP plots require two dimensions")
    coordinates = umap.UMAP(**dict(projection, n_components=2)).fit_transform(
        embeddings
    )
    label_column = "Aromaticity" if args.panel == "aromaticity" else "Condensed"
    positive = settings["svm"]["positive_label"]
    classes = sorted(frame[label_column].unique())
    if positive not in classes:
        raise ValueError("Configured positive_label is absent from the cohort")
    negative = next(label for label in classes if label != positive)
    labels = np.where(frame[label_column].to_numpy() == positive, 1, -1)
    train = frame.split.to_numpy() == "train"
    scale = StandardScaler().fit(embeddings[train])
    standardized = scale.transform(embeddings)
    margin = float(settings["svm"]["C"])
    classifier = SVC(kernel="linear", C=margin).fit(standardized[train], labels[train])
    predicted, score, report, matrix, auroc, average_precision = evaluate_on_test(
        classifier,
        standardized[~train],
        labels[~train],
        (str(negative), str(positive)),
    )
    diagnostics = linear_separability_diagnostics(
        standardized[train], labels[train], classifier=classifier
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projected = frame.copy()
    projected["0"], projected["1"] = coordinates[:, 0], coordinates[:, 1]
    filename = (
        "interp_benzene.csv" if args.panel == "aromaticity" else "interp_condensed.csv"
    )
    projected.to_csv(args.output_dir / filename, index=False)
    pd.DataFrame(
        {
            "molecule_id": frame.loc[~train, "molecule_id"],
            "target": labels[~train],
            "prediction": predicted,
            "score": score,
        }
    ).to_csv(args.output_dir / "svm-validation.csv", index=False)
    metrics = {
        "validation_auroc": float(auroc),
        "validation_average_precision": float(average_precision),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "training_margin_diagnostics": {
            key: float(value) for key, value in diagnostics.items()
        },
    }
    (args.output_dir / "svm-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tokens = sub.add_parser("tokens")
    tokens.add_argument("--model", type=Path, required=True)
    tokens.add_argument(
        "--fallback-tokenizer",
        type=Path,
        help="Matching base-model tokenizer JSON, used when the head omits it",
    )
    tokens.add_argument("--output-dir", type=Path, required=True)
    for name in ("encode", "analyze"):
        command = sub.add_parser(name)
        command.add_argument(
            "--panel", choices=["aromaticity", "condensation"], required=True
        )
        command.add_argument("--cohort", type=Path, required=True)
        if name == "encode":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--batch-size", type=int, default=8)
        else:
            command.add_argument("--embeddings", type=Path, required=True)
            command.add_argument("--settings", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "tokens":
        token_weights(
            args.model, args.output_dir, fallback_tokenizer=args.fallback_tokenizer
        )
    elif args.command == "encode":
        encode(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
