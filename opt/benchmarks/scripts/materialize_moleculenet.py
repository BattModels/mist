"""Materialize the repository's canonical MoleculeNet split as CSV files."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import hashlib
import json
import shutil
import ssl
import urllib.request
from pathlib import Path

import truststore
from datasets import load_dataset

from electrolyte_fm.data_modules.molnet_dataset import _URLS
from electrolyte_fm.data_modules.utils import scaffold_split, train_val_test_split

from .canonical_tasks import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-definitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    definitions = load(args.task_definitions)
    task = definitions[args.dataset]
    source_url = _URLS[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".csv.gz" if source_url.endswith(".gz") else ".csv"
    source = args.output_dir / f"source{suffix}"
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with (
        urllib.request.urlopen(source_url, context=context) as response,
        source.open("wb") as destination,
    ):
        shutil.copyfileobj(response, destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset = load_dataset(
        "csv",
        name=args.dataset,
        data_files=[str(source)],
        split="train",
        keep_in_memory=False,
    )

    if args.dataset == "qm8":
        dataset = dataset.rename_columns(
            {
                "E1-CC2": "E1-CC2-RI-CC2/def2TZVP",
                "E2-CC2": "E2-CC2-RI-CC2/def2TZVP",
                "f1-CC2": "f1-CC2-RI-CC2/def2TZVP",
                "f2-CC2": "f2-CC2-RI-CC2/def2TZVP",
                "E1-PBE0": "E1-PBE0-LR-TDPBE0/def2SVP",
                "E2-PBE0": "E2-PBE0-LR-TDPBE0/def2SVP",
                "f1-PBE0": "f1-PBE0-LR-TDPBE0/def2SVP",
                "f2-PBE0": "f2-PBE0-LR-TDPBE0/def2SVP",
                "E1-PBE0.1": "E1-PBE0-LR-TDPBE0/def2TZVP",
                "E2-PBE0.1": "E2-PBE0-LR-TDPBE0/def2TZVP",
                "f1-PBE0.1": "f1-PBE0-LR-TDPBE0/def2TZVP",
                "f2-PBE0.1": "f2-PBE0-LR-TDPBE0/def2TZVP",
                "E1-CAM": "E1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                "E2-CAM": "E2-CAM-LR-TDCAM-B3LYP/def2TZVP",
                "f1-CAM": "f1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                "f2-CAM": "f2-CAM-LR-TDCAM-B3LYP/def2TZVP",
            }
        )

    smiles_column = "mol" if args.dataset == "bace" else "smiles"
    if smiles_column != "smiles":
        dataset = dataset.rename_column(smiles_column, "smiles")
        smiles_column = "smiles"
    split_name = task["split"]
    if split_name == "scaffold":
        splits = scaffold_split(dataset, smiles_column)
    elif split_name == "random":
        splits = train_val_test_split(dataset)
    else:
        raise ValueError(f"unsupported split: {split_name}")

    targets = task["target_columns"]
    required = {"smiles", *targets}
    counts = {}
    for name in ("train", "validation", "test"):
        split = splits[name]
        missing = required - set(split.column_names)
        if missing or not len(split):
            raise ValueError(f"{name}: missing={sorted(missing)}, rows={len(split)}")
        split.to_csv(args.output_dir / f"{name}.csv", index=False)
        counts[name] = len(split)

    protocol = {
        "dataset": args.dataset,
        "source_url": source_url,
        "source_sha256": digest,
        "split": split_name,
        "seed": 42,
        "counts": counts,
        "targets": targets,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
