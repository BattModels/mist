"""Reproduce the prepared seed-42 AEM split from the pinned public table."""

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd

SOURCE_SHA256 = "2f44ba6e69c07bbee4c914afa5b47ad9ae2b0769eb951b22330f4ecf3a244c7f"
SOLVENTS = {
    "Propylene Carbonate": "CC1COC(=O)O1",
    "Fluoroethylene Carbonate": "O=C1OCC(F)O1",
    "Ethylene Carbonate": "O=C1OCCO1",
    "Ethyl Methyl Carbonate": "CCOC(=O)OC",
    "Diethyl Carbonate": "CCOC(=O)OCC",
    "Dimethyl Carbonate": "COC(=O)OC",
}
ANIONS = {
    "LiPF6": "F[P-](F)(F)(F)(F)F",
    "LiTFSI": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
}


def prepare(source, output):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Public AEM source does not match the pinned table")
    data = pd.read_csv(source)
    splits = {name: [] for name in ("train", "val", "test")}
    for _, row in data.iterrows():
        # Preserve column order and float rendering: both define the existing
        # prepared split's hash. This is a replacement split, not the paper split.
        values = {
            "smi1": SOLVENTS[row.sub1_name], "x1": row["x1 (mole fraction)"],
            "smi2": SOLVENTS[row.sub2_name], "x2": row["x2 (mole fraction)"],
            "smi3": SOLVENTS[row.sub3_name], "x3": row["x3 (mole fraction)"],
            "smi4": ANIONS[row.salt], "x4": row["x4 (anion, mole fraction)"],
            "smi5": "[Li+]", "x5": row["x5 (cation, mole fraction)"],
            "temperature": row["temperature (K)"],
            "ln k": math.log(row["ionic conductivity (mS/cm)"]),
        }
        key = "|".join(str(value) for value in values.values())
        bucket = int(hashlib.sha256(("seed42|" + key).encode()).hexdigest()[:16], 16) % 10
        split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
        splits[split].append(values)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pd.DataFrame(rows).to_csv(output / f"{split}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output_dir)


if __name__ == "__main__":
    main()
