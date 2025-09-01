#!/usr/bin/env -S uv run python
"""
Process CLC-DB molecules.csv into a HuggingFace datasets Arrow dataset.

Steps:
- Read the CSV produced by export_clc_db.py
- Validate SMILES (RDKit); drop invalid/empty
- Compute InChI and InChIKey (RDKit)
- Convert "Chirality" to multi-label dict with keys: point, planar, axial
- Split into train/val/test (80/10/10) using StratifiedShuffleSplit, stratify on chirality
- Save as a DatasetDict via datasets.save_to_disk

Usage:
  uv run opt/sterochemistry/process_clc_db.py \
    --csv opt/sterochemistry/data/clc_db/molecules.csv \
    --out opt/sterochemistry/data/clc_db/hf
"""

import argparse
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
from sklearn.model_selection import StratifiedShuffleSplit


# Column names from the website/export script
COL_SMILES = "SMILES"
COL_CHIRALITY = "Chirality"
COL_MW = "Molecular Weight"
COL_ZP = "Zero-point correction"
COL_TE = "Thermal correction to Energy"
COL_TH = "Thermal correction to Enthalpy"
COL_TG = "Thermal correction to Gibbs Free Energy"
COL_HOMO = "HOMO Energy (eV)"
COL_LUMO = "LUMO Energy (eV)"
COL_GAP = "HOMO-LUMO Gap (eV)"


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isfinite(x):
            return float(x)
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "n/a", "nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def parse_chirality(raw: str | None) -> dict[str, bool]:
    """Parse the website's chirality string into multi-label booleans.

    - Recognizes: point, planar, axial (case-insensitive, substring match)
    - Treats N/A or empty as all False
    - If multiple classes are present, sets both True
    """
    flags = {"point": False, "planar": False, "axial": False}
    if not raw:
        return flags
    s = str(raw).strip()
    if s == "" or s.lower() in {"na", "n/a", "none"}:
        return flags
    s_low = s.lower()
    if "point" in s_low:
        flags["point"] = True
    if "planar" in s_low:
        flags["planar"] = True
    if "axial" in s_low:
        flags["axial"] = True
    return flags


def smiles_to_mol(smiles: str | None):
    if not smiles:
        return None
    s = smiles.strip()
    if s == "":
        return None
    try:
        m = Chem.MolFromSmiles(s)
        return m
    except Exception:
        return None


def compute_inchi(mol) -> tuple[str | None, str | None]:
    if mol is None:
        return None, None
    try:
        inchi = MolToInchi(mol)
    except Exception:
        inchi = None
    try:
        ikey = MolToInchiKey(mol)
    except Exception:
        ikey = None
    return inchi, ikey


def load_rows(csv_path: str) -> list[dict[str, Any]]:
    # Read all columns as strings to avoid Pandas NA coercion, then convert
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        smiles = r.get(COL_SMILES, "").strip()
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue  # drop invalid/empty SMILES
        inchi, inchi_key = compute_inchi(mol)

        ch = parse_chirality(r.get(COL_CHIRALITY, ""))

        row: dict[str, Any] = {
            "smiles": smiles,
            "InChI": inchi if inchi is not None else "",
            "InChIKey": inchi_key if inchi_key is not None else "",
            "chirality": ch,
            "molecular_weight": _to_float(r.get(COL_MW)),
            "zero_point_correction": _to_float(r.get(COL_ZP)),
            "thermal_correction_energy": _to_float(r.get(COL_TE)),
            "thermal_correction_enthalpy": _to_float(r.get(COL_TH)),
            "thermal_correction_gibbs": _to_float(r.get(COL_TG)),
            "homo": _to_float(r.get(COL_HOMO)),
            "lumo": _to_float(r.get(COL_LUMO)),
            "gap": _to_float(r.get(COL_GAP)),
        }
        rows.append(row)
    return rows


def build_multilabel_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    y = np.zeros((len(rows), 3), dtype=int)
    for i, r in enumerate(rows):
        ch = r.get("chirality", {}) or {}
        y[i, 0] = 1 if ch.get("point", False) else 0
        y[i, 1] = 1 if ch.get("planar", False) else 0
        y[i, 2] = 1 if ch.get("axial", False) else 0
    return y


def _stratify_indices(
    n: int, y: np.ndarray, seed: int = 17
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test (80/10/10) using stratification on y.

    Primary path: pass multi-label indicator matrix directly to StratifiedShuffleSplit
    as requested. If the environment's sklearn does not support this, fall back to
    stratifying on label-combinations as a single multiclass target.
    """
    rng = seed

    def _try_split(y_input) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_all = np.arange(n)
        # first split off test 10%
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=rng)
        train_val_idx, test_idx = next(sss1.split(idx_all, y_input))
        # then split train/val with val proportion 1/9 (~0.111... of remaining)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=rng)
        train_idx, val_idx = next(sss2.split(train_val_idx, y_input[train_val_idx]))
        return train_val_idx[train_idx], train_val_idx[val_idx], test_idx

    # Attempt multi-label directly
    try:
        return _try_split(y)
    except Exception:
        # Fall back to combination labels as strings
        combos = np.array([f"{a}{b}{c}" for a, b, c in y], dtype=object)
        return _try_split(combos)


def to_hf_datasets(
    rows: list[dict[str, Any]], indices: dict[str, np.ndarray]
) -> DatasetDict:
    features = Features(
        {
            "smiles": Value("string"),
            "InChI": Value("string"),
            "InChIKey": Value("string"),
            "chirality": {
                "point": Value("bool"),
                "planar": Value("bool"),
                "axial": Value("bool"),
            },
            "molecular_weight": Value("float64"),
            "zero_point_correction": Value("float64"),
            "thermal_correction_energy": Value("float64"),
            "thermal_correction_enthalpy": Value("float64"),
            "thermal_correction_gibbs": Value("float64"),
            "homo": Value("float64"),
            "lumo": Value("float64"),
            "gap": Value("float64"),
        }
    )

    def _subset(split_idx: np.ndarray) -> Dataset:
        data = [rows[i] for i in split_idx]
        return Dataset.from_list(data, features=features)

    dsd = DatasetDict(
        {
            "train": _subset(indices["train"]),
            "validation": _subset(indices["validation"]),
            "test": _subset(indices["test"]),
        }
    )
    return dsd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default=os.path.join("data", "clc_db", "molecules.csv"),
        help="Input molecules.csv path",
    )
    ap.add_argument(
        "--out",
        default=os.path.join("data", "clc_db", "hf"),
        help="Output directory for the saved DatasetDict",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    if not rows:
        print("No valid molecules found after SMILES validation.")
        return 1

    y = build_multilabel_targets(rows)
    train_idx, val_idx, test_idx = _stratify_indices(len(rows), y, seed=args.seed)
    indices = {"train": train_idx, "validation": val_idx, "test": test_idx}

    dsd = to_hf_datasets(rows, indices)
    os.makedirs(args.out, exist_ok=True)
    dsd.save_to_disk(args.out, num_shards={k: 4 for k in indices.keys()})

    # Brief summary
    print(
        json.dumps(
            {
                "counts": {
                    "train": len(train_idx),
                    "validation": len(val_idx),
                    "test": len(test_idx),
                    "total": len(rows),
                },
                "out": os.path.abspath(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
