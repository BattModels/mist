"""Prepare the available Wolfram isotope cohort with the original parser/splitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset

from electrolyte_fm.data_modules.utils import train_val_test_split

from opt.isotopes.preprocess_isotopes import read_isotopes

TARGETS = ["half_life", "log_time"]


def materialize(
    source: Path, output: Path, *, sha256: str | None = None,
    source_url: str | None = None,
) -> None:
    if not source.is_file():
        if source_url is None:
            raise FileNotFoundError(f"Missing isotope source at {source}; supply --source-url to download it")
        if sha256 is None:
            raise ValueError("Downloading the isotope source requires --sha256")
        subprocess.run(
            [sys.executable, "-m", "electrolyte_fm.utils.fetch_file",
             source_url, source, "--sha256", sha256],
            check=True,
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"Isotope source SHA256 mismatch: expected {sha256}, got {digest}")
    cohort = read_isotopes(source)
    if cohort.empty:
        raise ValueError("The isotope cohort is empty")
    if not np.isfinite(cohort[TARGETS].to_numpy(float)).all() or (cohort.half_life <= 0).any():
        raise ValueError("The original half-life coefficient must be positive and both targets finite")
    cohort.insert(0, "row_id", np.arange(len(cohort)))
    splits = train_val_test_split(Dataset.from_pandas(cohort, preserve_index=False))
    membership = {row_id: name for name, split in splits.items() for row_id in split["row_id"]}
    cohort["split"] = cohort.row_id.map(membership)
    payloads = {"cohort.csv": cohort.to_csv(index=False)}
    for name, split in splits.items():
        payloads[f"{name}.csv"] = split.to_pandas().to_csv(index=False)
    payloads["protocol.json"] = json.dumps({
        "dataset": "isotopes", "source_file": source.name, "source_sha256": digest,
        "rows": len(cohort), "counts": {name: len(split) for name, split in splits.items()},
        "unique_smiles": int(cohort.smiles.nunique()),
        "retained_higher_metastable_rows": int(cohort.name.str.contains(r"m\d+$").sum()),
        "shared_smiles_between_splits": {
            f"{left}/{right}": len(set(splits[left]["smiles"]) & set(splits[right]["smiles"]))
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
        },
        "filter": "original parser: omit names ending in m, m1 or m2; preserve source order",
        "split": "train_val_test_split: seeded 80/10/10", "seed": 42,
        "targets": TARGETS,
        "target_meanings": {"half_life": "scaled half-life coefficient", "log_time": "decimal time-scale exponent"},
        "half_life_seconds": "half_life * 10**log_time",
    }, indent=2, sort_keys=True) + "\n"
    # Never overwrite different established cohort/split contents on a retry.
    for filename, payload in payloads.items():
        path = output / filename
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Existing isotope artifact differs: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        path = output / filename
        if not path.exists():
            path.write_text(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-url", help="Download a missing source from this URL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    materialize(args.source, args.output_dir, sha256=args.sha256, source_url=args.source_url)


if __name__ == "__main__":
    main()
