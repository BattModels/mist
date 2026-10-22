"""Prepare shared, reproducible PAH cohorts from pinned public COMPAS tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.model_selection import train_test_split


def canonical_and_aromatic(smiles: str) -> tuple[str, bool]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected SMILES: {smiles}")
    molecule = Chem.RemoveHs(molecule)
    bonds = list(molecule.GetBonds())
    if not bonds:
        raise ValueError(f"Molecule has no bonds: {smiles}")
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        all(bond.GetIsAromatic() for bond in bonds),
    )


def sample_and_split(rows, label, per_class, seed, validation_fraction):
    rng = np.random.default_rng(seed)
    sampled = []
    groups = pd.DataFrame(rows).groupby(label, sort=True)
    if len(groups) != 2:
        raise ValueError(f"The {label} cohort must contain both chemical classes")
    for _, group in groups:
        if len(group) < per_class:
            raise ValueError(
                f"Requested {per_class} rows from a class with {len(group)}"
            )
        sampled.append(
            group.iloc[np.sort(rng.choice(len(group), per_class, replace=False))]
        )
    frame = pd.concat(sampled, ignore_index=True)
    _, validation = train_test_split(
        frame.index,
        test_size=validation_fraction,
        random_state=seed,
        stratify=frame[label],
    )
    frame["split"] = "train"
    frame.loc[validation, "split"] = "validation"
    return frame


def prepare_cohorts(
    compas1: Path,
    compas2: Path,
    compas3: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    per_class: int = 1000,
    validation_fraction: float = 0.2,
) -> None:
    c1 = pd.read_csv(compas1, usecols=["molecule", "smiles", "n_rings"])
    c2 = pd.read_parquet(
        compas2, columns=["name", "smiles", "charge", "rings", "benzene"]
    )
    c3 = pd.read_csv(compas3, usecols=["molecule", "smiles", "n_rings"])
    aromaticity, seen = [], set()
    # One neutral representative per graph avoids repeating the three charge states.
    for row in c2.loc[c2.charge == 0].itertuples(index=False):
        canonical, aromatic = canonical_and_aromatic(row.smiles)
        if canonical in seen:
            continue
        seen.add(canonical)
        aromaticity.append(
            {
                "molecule_id": "COMPAS-2D:" + str(row.name),
                "source_dataset": "COMPAS-2D",
                "source_id": row.name,
                "smiles": canonical,
                "n_rings": int(row.rings),
                "Number of Benzene": int(row.benzene),
                "Aromaticity": "Aromatic" if aromatic else "Not fully aromatic",
            }
        )

    condensation, graphs = [], {}
    for label, source, frame in [("Cata", "COMPAS-1D", c1), ("Peri", "COMPAS-3D", c3)]:
        graphs[label] = set()
        for row in frame.itertuples(index=False):
            canonical, _ = canonical_and_aromatic(row.smiles)
            if canonical in graphs[label]:
                continue
            graphs[label].add(canonical)
            condensation.append(
                {
                    "molecule_id": source + ":" + str(row.molecule),
                    "source_dataset": source,
                    "source_id": row.molecule,
                    "smiles": canonical,
                    "n_rings": int(row.n_rings),
                    "Condensed": label,
                }
            )
    # Shared structures cannot be assigned both dataset class labels.
    overlap = graphs["Cata"] & graphs["Peri"]
    condensation = [row for row in condensation if row["smiles"] not in overlap]

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for panel, rows, label, positive in [
        ("aromaticity", aromaticity, "Aromaticity", "Aromatic"),
        ("condensation", condensation, "Condensed", "Peri"),
    ]:
        cohort = sample_and_split(rows, label, per_class, seed, validation_fraction)
        cohort.to_csv(output_dir / (panel + ".csv"), index=False)
        settings = {
            "umap": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": seed,
                "transform_seed": seed,
            },
            "svm": {"C": 100.0, "positive_label": positive},
        }
        (output_dir / (panel + "-settings.json")).write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        counts[panel] = {
            "candidates": pd.DataFrame(rows)[label].value_counts().to_dict(),
            "selected": cohort[label].value_counts().to_dict(),
            "splits": cohort.split.value_counts().to_dict(),
        }

    source = json.loads(Path(__file__).with_name("compas.json").read_text())
    hashes = {}
    for key, path in [("compas1", compas1), ("compas2", compas2), ("compas3", compas3)]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[key] = digest.hexdigest()
    provenance = {
        "source": source,
        "input_sha256": hashes,
        "rdkit_version": rdBase.rdkitVersion,
        "cohort_policy": "New deterministic samples; original paper sample identities were not released",
        "seed": seed,
        "per_class": per_class,
        "validation_fraction": validation_fraction,
        "aromaticity_rule": "All heavy-atom bonds have RDKit GetIsAromatic() == True",
        "charge_policy": "Neutral COMPAS-2D records only",
        "condensation_rule": "COMPAS-1D Cata / COMPAS-3D Peri labels",
        "deduplication": "Canonical isomeric SMILES; cross-class condensation overlaps excluded",
        "condensation_overlap_removed": len(overlap),
        "counts": counts,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["compas1", "compas2", "compas3", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    prepare_cohorts(
        args.compas1,
        args.compas2,
        args.compas3,
        args.output_dir,
        seed=args.seed,
        per_class=args.per_class,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
