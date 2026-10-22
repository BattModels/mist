"""Materialize the pinned OpenPOM 4,983-row public olfaction replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.source)
    channels = [
        item["name"] for item in yaml.safe_load(args.schema.read_text())["channels"]
    ]
    if len(channels) != 135 or not set(channels).issubset(frame.columns):
        raise ValueError(
            "OpenPOM does not match the ordered 135-channel odour contract"
        )
    frame = frame.rename(columns={"nonStereoSMILES": "smiles"})[["smiles", *channels]]
    shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    train_end = int(len(frame) * 0.8)
    validation_end = int(len(frame) * 0.9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in {
        "train": shuffled.iloc[:train_end],
        "validation": shuffled.iloc[train_end:validation_end],
        "test": shuffled.iloc[validation_end:],
    }.items():
        part.to_csv(args.output_dir / f"{name}.csv", index=False)
    protocol = {
        "status": "public_replacement_not_asserted_exact_paper_split",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "rows": len(frame),
        "targets": channels,
        "split": "seed-42 random 80/10/10",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
