"""Extract isotope embeddings from a selected local task model and cohort."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_cohort(path: Path, *, expected_rows: int | None = None) -> pd.DataFrame:
    from opt.isotopes.preprocess_isotopes import get_protons_neutrons

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing isotope cohort at {path}; run benchmark-isotopes-materialize."
        )
    frame = pd.read_csv(path)
    columns = ["smiles", "neutrons", "protons"]
    if not set(columns) <= set(frame) or frame[columns].isna().any().any():
        raise ValueError(f"{path} requires nonempty columns {columns}")
    frame = frame[columns].copy()
    if frame.empty:
        raise ValueError("The isotope cohort is empty")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} isotopes, got {len(frame)}")
    if not all(isinstance(value, str) and re.fullmatch(r"\[\d+[A-Z][a-z]?\]", value) for value in frame.smiles):
        raise ValueError("Every cohort SMILES must be one nonmetastable [massElement] isotope")
    counts = frame.apply(get_protons_neutrons, axis=1)
    for column in ["neutrons", "protons"]:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        if not np.array_equal(values, counts[column].to_numpy()):
            raise ValueError(f"{column} must agree with the isotope mass and element in smiles")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {column} values")
        frame[column] = values.astype(int)
    return frame


def read_protocol(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing isotope embedding/projection settings at {path}; "
            "the workflow uses opt/isotopes/protocol.json."
        )
    protocol = json.loads(path.read_text())
    required = {"encoding", "pooling", "projection_columns", "n_neighbors", "min_dist", "metric", "seed",
                "tol_offset", "tol_scale", "umap_kwargs"}
    if not isinstance(protocol, dict) or not required <= protocol.keys():
        raise ValueError(f"Isotope protocol must supply: {', '.join(sorted(required))}")
    if protocol["encoding"] not in {"smiles", "smiles-kekule", "smiles-canonical"}:
        raise ValueError("encoding must identify the original SMILES representation")
    if protocol["pooling"] != "cls":
        raise ValueError("The existing MIST task embedding method supports first-token pooling (cls)")
    columns = protocol["projection_columns"]
    if not (columns in ("embeddings", "legacy_numeric") or
            isinstance(columns, list) and columns and all(isinstance(c, str) for c in columns)):
        raise ValueError("projection_columns must be embeddings, legacy_numeric, or an ordered list of CSV columns")
    if protocol["metric"] not in {"manhattan", "cosine", "euclidean"}:
        raise ValueError("metric must be manhattan, cosine, or euclidean")
    if type(protocol["seed"]) is not int or type(protocol["n_neighbors"]) is not int or protocol["n_neighbors"] < 2:
        raise ValueError("seed and n_neighbors must be integers; n_neighbors must be at least 2")
    for key in ["min_dist", "tol_offset", "tol_scale"]:
        if not isinstance(protocol[key], (int, float)) or not np.isfinite(protocol[key]) or protocol[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    kwargs = protocol["umap_kwargs"]
    if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
        raise ValueError("umap_kwargs must be a JSON object of original UMAP.jl keyword overrides")
    if set(kwargs) & {"n_neighbors", "min_dist", "metric"}:
        raise ValueError("Set n_neighbors, min_dist and metric at the protocol's top level")
    return protocol


def encode(args) -> None:
    frame = read_cohort(args.cohort)
    protocol = read_protocol(args.protocol)
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing selected isotope MIST model directory at {args.model}; "
            "run the benchmark isotope LR sweep and validation selection/test task."
        )
    import torch

    from electrolyte_fm.data_modules.utils import MolEncoding
    from electrolyte_fm.models.prod_finetune import load_finetuned

    smiles = [MolEncoding(protocol["encoding"])(value) for value in frame.smiles]
    if any(value is None for value in smiles):
        raise ValueError("The isotope cohort contains an invalid SMILES; no rows are dropped")
    model = load_finetuned(str(args.model), local_files_only=True).eval().to(args.device)
    with torch.inference_mode():
        values = model.embed(smiles, batch_size=args.batch_size).float().cpu().numpy()
    if values.shape != (len(frame), model.encoder.config.hidden_size) or not np.isfinite(values).all():
        raise ValueError("Unexpected isotope embedding shape or nonfinite values")
    embeddings = pd.DataFrame(values, columns=[f"embedding_{index}" for index in range(values.shape[1])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame, embeddings], axis=1).to_csv(args.output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["cohort", "model", "protocol", "output"]:
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    encode(parser.parse_args())


if __name__ == "__main__":
    main()
