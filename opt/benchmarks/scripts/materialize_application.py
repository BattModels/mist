"""Preserve established application splits; prepare new local inputs without downloads."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from rdkit import Chem

from electrolyte_fm.data_modules.utils import train_val_test_split

BP_REVISION = "55629ff3591ec4d666ef4f22c59a685a90ba09f2"
MP_REVISION = "35a324d6a9ad5766b5c78d0d5b0591e7569d5921"
OPENPOM_REVISION = "5434c4b7badebca8e7d180aa34d2f787402121ee"
ESTABLISHED_TASKS = ("bp", "mp", "odour", "pka", "etn", "kt")
TARGETS = {
    "bp": ["bp"],
    "mp": ["mp"],
    "fp": ["fp"],
    "dn": ["BF3 affinity"],
    "pka": ["DMSO_pKa"],
    "etn": ["ETN"],
    "kt": ["alpha", "beta"],
}


def require_cached(path: Path) -> Path:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(
            f"Restore the original source at {path}; no download fallback is used."
        )
    return path


def source_record(path: Path, url: str) -> dict:
    return {"path": str(path.resolve()), "url": url}


def temperature(task: str, cache: Path) -> tuple[list[dict], list[dict]]:
    sources, records = [], []
    if task == "bp":
        for part in ("training", "prediction"):
            name = f"BP_{part}.jsonl"
            url = f"https://raw.githubusercontent.com/YuxuanLongBeyond/GNN-BP/{BP_REVISION}/data/bp/{name}"
            path = require_cached(cache / name)
            sources.append(source_record(path, url))
            records.extend(
                {"smiles": row["smiles"], "bp": row["Tox"]}
                for row in map(json.loads, path.read_text().splitlines())
            )
    else:
        url = f"https://raw.githubusercontent.com/nadinulrich/MP_prediction/{MP_REVISION}/Data/data_annotated.csv"
        path = require_cached(cache / "mp_data_annotated.csv")
        sources.append(source_record(path, url))
        with path.open(encoding="cp1252", newline="") as stream:
            records = [
                {
                    "smiles": row["canoncial SMILES"] or row["original SMILES"],
                    "mp": row["MP"],
                }
                for row in csv.DictReader(stream, delimiter=";")
                if row["excluded or corrected"].strip().lower() != "x"
            ]
    return records, sources


def pka(cache: Path) -> tuple[list[dict], list[dict]]:
    url = "https://zenodo.org/records/16882613/files/pKa_Data_Table.xlsx?download=1"
    path = cache / "pKa_Data_Table-v1.4.0.csv"
    if path.is_file():
        frame = pd.read_csv(path)
    else:
        path = require_cached(cache / "pKa_Data_Table-v1.4.0.xlsx")
        frame = pd.read_excel(path)
    records = frame.rename(columns={"Canonical_SMILES": "smiles"}).to_dict("records")
    return records, [source_record(path, url)]


def solvent_table(task: str, cache: Path) -> tuple[list[dict], list[dict]]:
    url = "https://www.stenutz.eu/chem/" + (
        "solv20.php?sort=3" if task == "etn" else "solv26.php"
    )
    snapshot = require_cached(cache / f"stenutz-{task}-snapshot.json")
    return json.loads(snapshot.read_text())["rows"], [source_record(snapshot, url)]


def curate(task: str, records: list[dict]) -> list[dict]:
    targets = TARGETS[task]
    groups = defaultdict(list)
    smiles_by_key = {}
    for record in records:
        smiles = record.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        if task in ("bp", "mp"):
            if molecule.GetNumHeavyAtoms() >= 20 or any(
                atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
            ):
                continue
        elif len(Chem.GetMolFrags(molecule)) != 1:
            continue
        try:
            values = [float(record[target]) for target in targets]
        except (ValueError, TypeError, KeyError):
            continue
        if not all(map(math.isfinite, values)):
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        key = record.get("InChIKey") if task == "pka" else canonical
        if not isinstance(key, str) or not key.strip():
            continue
        groups[key.strip()].append(values)
        smiles_by_key[key.strip()] = canonical
    rows = [
        {
            "smiles": smiles_by_key[key],
            **dict(zip(targets, np.mean(values, axis=0), strict=True)),
        }
        for key, values in sorted(groups.items())
    ]
    if len(rows) < {"bp": 5000, "mp": 13000, "pka": 3000, "etn": 300, "kt": 150}[task]:
        raise ValueError(
            f"{task}: only {len(rows)} curated rows; inspect source/parser before training"
        )
    return rows


def read_cached_source(
    task: str, cache: Path, odour_schema: Path | None = None
) -> tuple[list[dict], list[dict]]:
    """Use precisely the sources and curation that produced the completed runs."""
    if task == "odour":
        if odour_schema is None:
            raise ValueError("The odour dataset requires --odour-schema")
        url = f"https://raw.githubusercontent.com/BioMachineLearning/openpom/{OPENPOM_REVISION}/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
        path = require_cached(cache / "curated_GS_LF_merged_4983.csv")
        channels = [
            item["name"]
            for item in yaml.safe_load(odour_schema.read_text())["channels"]
        ]
        frame = pd.read_csv(path).rename(columns={"nonStereoSMILES": "smiles"})
        if len(channels) != 135 or len(frame) != 4983:
            raise ValueError(
                "OpenPOM does not match its 4,983-row, 135-channel contract"
            )
        return frame[["smiles", *channels]].to_dict("records"), [
            source_record(path, url)
        ]
    if task in ("bp", "mp"):
        records, sources = temperature(task, cache)
    elif task == "pka":
        records, sources = pka(cache)
    elif task in ("etn", "kt"):
        records, sources = solvent_table(task, cache)
    else:
        raise ValueError(f"No established source for {task}; supply --source")
    return curate(task, records), sources


def read_source(
    task: str, source: Path, odour_schema: Path | None = None
) -> tuple[list[dict], list[dict]]:
    if not source.is_file():
        raise FileNotFoundError(
            f"Supply the {task} dataset at {source}. "
            "Input files must be placed locally; no download fallback is used."
        )
    frame = pd.read_csv(source, encoding="utf-8-sig")
    targets = TARGETS.get(task)
    if task == "odour":
        if odour_schema is None:
            raise ValueError("The odour dataset requires --odour-schema")
        targets = [
            item["name"]
            for item in yaml.safe_load(odour_schema.read_text())["channels"]
        ]
        if len(targets) != 135 or len(frame) != 4983:
            raise ValueError("OpenPOM requires its 4,983-row, 135-channel source")
    if targets is None:
        raise ValueError(f"Unknown application dataset: {task}")

    # Support the supplied CSV headers without rewriting the source files.
    renames = {}
    if "smiles" not in frame:
        aliases = [
            name
            for name in ("iso_smiles", "smi", "nonStereoSMILES", "Canonical_SMILES")
            if name in frame
        ]
        if len(aliases) == 1:
            renames[aliases[0]] = "smiles"
    if task == "etn" and "ETN" not in frame and "E T N (num)" in frame:
        renames["E T N (num)"] = "ETN"
    frame = frame.rename(columns=renames)
    missing = {"smiles", *targets} - set(frame.columns)
    if missing:
        detail = (
            " Add verified SMILES to the supplied name-only table; "
            "names are not resolved online or silently dropped."
            if task == "kt" and "smiles" in missing
            else ""
        )
        raise ValueError(f"{source}: missing columns {sorted(missing)}.{detail}")
    frame = frame[["smiles", *targets]].copy()
    if frame.empty or frame.smiles.isna().any():
        raise ValueError(f"{source}: expected nonempty rows with SMILES")
    for target in targets:
        frame[target] = pd.to_numeric(frame[target], errors="raise")
        if np.isinf(frame[target]).any() or not frame[target].notna().any():
            raise ValueError(f"{source}: invalid or entirely missing {target} values")

    # The original CSV datamodule splits before encoding/filtering. Preserve
    # membership: do not sort, average duplicates, or drop invalid SMILES here.
    return frame.to_dict("records"), [
        {"path": str(source.resolve()), "column_renames": renames}
    ]


def split_frames(task: str, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # These are the exact pre-existing algorithms, including within-split order.
    # A common seed does NOT make different shuffle implementations equivalent.
    if task in ("bp", "mp"):
        order = list(range(len(frame)))
        random.Random(42).shuffle(order)
        end = int(0.8 * len(frame))
        val_end = end + int(0.1 * len(frame))
        membership = {
            "train": order[:end],
            "validation": order[end:val_end],
            "test": order[val_end:],
        }
    elif task == "odour":
        order = frame.sample(frac=1, random_state=42).index.tolist()
        end, val_end = int(0.8 * len(frame)), int(0.9 * len(frame))
        membership = {
            "train": order[:end],
            "validation": order[end:val_end],
            "test": order[val_end:],
        }
    elif task in ("pka", "etn", "kt"):
        order = np.random.default_rng(42).permutation(len(frame))
        count = math.ceil(0.1 * len(frame))
        membership = {
            "test": sorted(order[:count]),
            "validation": sorted(order[count : 2 * count]),
            "train": sorted(order[2 * count :]),
        }
    else:
        if task not in ("fp", "dn"):
            raise ValueError(f"Unknown application dataset: {task}")
        splits = train_val_test_split(Dataset.from_pandas(frame, preserve_index=False))
        return {name: split.to_pandas() for name, split in splits.items()}
    return {name: frame.iloc[indices] for name, indices in membership.items()}


def write_splits(
    task: str, rows: list[dict], sources: list[dict], output: Path
) -> None:
    frame = pd.DataFrame(rows)
    splits = split_frames(task, frame)
    contents = {
        output / f"{name}.csv": split.to_csv(index=False).encode("utf-8")
        for name, split in splits.items()
    }
    # Preflight every split before writing anything. Existing CSVs are immutable,
    # including row order, duplicate measurements, formatting, and timestamps.
    for path, expected in contents.items():
        if path.exists() and path.read_bytes() != expected:
            raise ValueError(
                f"{path}: existing split content or row order differs; refusing to overwrite"
            )
    established = task in ESTABLISHED_TASKS
    protocol = {
        "dataset": task,
        "exact_paper_split": False,
        "status": (
            "public_replacement_not_asserted_exact_paper_split"
            if established
            else "user_supplied_source_shared_split"
        ),
        "sources": sources,
        "targets": list(frame.columns.drop("smiles")),
        "rows": len(frame),
        "split_seed": 42,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "selection": "validation only; one test evaluation after learning-rate selection",
    }
    if established:
        protocol["exact_paper_dataset"] = False
    else:
        protocol["source_row_order_preserved"] = True
        protocol["split_implementation"] = (
            "electrolyte_fm.data_modules.utils.train_val_test_split"
        )
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        for key in ("dataset", "rows", "targets", "split_seed", "split_counts"):
            if key in existing and existing[key] != protocol[key]:
                raise ValueError(
                    f"{protocol_path}: existing {key} differs; refusing to overwrite"
                )
    output.mkdir(parents=True, exist_ok=True)
    for path, content in contents.items():
        if not path.exists():
            path.write_bytes(content)
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*TARGETS, "odour"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--odour-schema", type=Path)
    args = parser.parse_args()
    if args.dataset in ESTABLISHED_TASKS:
        if args.source_dir is None:
            parser.error("Established datasets require their original --source-dir")
        rows, sources = read_cached_source(
            args.dataset, args.source_dir, args.odour_schema
        )
    else:
        if args.source is None:
            parser.error("New datasets require a local --source")
        rows, sources = read_source(args.dataset, args.source, args.odour_schema)
    write_splits(args.dataset, rows, sources, args.output_dir)


if __name__ == "__main__":
    main()
