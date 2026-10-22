"""Resolve all original OlfBoost observations and write a fresh seed-42 split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


def observations(definitions, annotations, identities):
    """Preserve every measured pair; zero and empty CID cells are padding."""
    if identities.CID.isna().any() or identities.CID.duplicated().any():
        raise ValueError("CID identities must be unique and nonempty")
    mapping = dict(zip(identities.CID, identities["Canonical SMILES"], strict=True))
    if any(not isinstance(smi, str) or Chem.MolFromSmiles(smi) is None
           for smi in mapping.values()):
        raise ValueError("CID mapping contains invalid molecular structures")
    keys = ["Dataset", "Mixture Label"]
    if definitions[keys].isna().any().any() or definitions.duplicated(keys).any():
        raise ValueError("Mixture definitions require unique dataset/label pairs")
    cid_columns = [name for name in definitions if name.startswith("CID_")]
    if not cid_columns:
        raise ValueError("Mixture definitions require CID_ component columns")
    mixtures = {}
    for _, row in definitions.iterrows():
        cids = [value for value in row[cid_columns].dropna() if value != 0]
        if not cids or any(value <= 0 or value != int(value) for value in cids):
            raise ValueError("Mixtures require positive integer component CIDs")
        missing = set(cids) - mapping.keys()
        if missing:
            raise ValueError(f"Missing molecular identities for CIDs {sorted(missing)}")
        mixtures[row.Dataset, row["Mixture Label"]] = json.dumps([mapping[cid] for cid in cids])
    rows = []
    for index, row in annotations.reset_index(drop=True).iterrows():
        rows.append({
            "observation_id": index,
            "dataset": row.Dataset,
            "mixture_1": row["Mixture 1"],
            "mixture_2": row["Mixture 2"],
            "mix1_smiles": mixtures[row.Dataset, row["Mixture 1"]],
            "mix2_smiles": mixtures[row.Dataset, row["Mixture 2"]],
            "value": row["Experimental Values"],
        })
    result = pd.DataFrame(rows)
    if result.empty or not np.isfinite(result.value).all() or not result.value.between(0, 1).all():
        raise ValueError("Experimental distances must be finite values in [0, 1]")
    return result


def materialize(args):
    frame = observations(*(pd.read_csv(path) for path in
                           (args.definitions, args.annotations, args.identities)))
    if len(frame) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} observations, found {len(frame)}")
    train, held_out = train_test_split(frame, test_size=0.2, random_state=args.seed)
    validation, test = train_test_split(held_out, test_size=0.5, random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", validation), ("test", test)):
        split.sort_values("observation_id").to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "split.json").write_text(json.dumps({
        "source": "CWYK-Olfboost original, unaugmented annotations",
        "seed": args.seed, "split_policy": "random-observation-80-10-10",
        "entity_disjoint": False, "zero_cid_policy": "padding",
        "observations": len(frame),
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("definitions", "annotations", "identities", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=780)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
