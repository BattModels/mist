"""Expose the recovered JCIM2017 Arrow corpus without copying its files."""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from electrolyte_fm.data_modules.pubchem_qc import DEFAULT_TOKEN_TARGETS
from electrolyte_fm.utils.stage_hf import link_model as link_directory


def materialize(dataset, output, provenance=None):
    dataset = Path(dataset).resolve()
    files = {
        name: sorted((dataset / name).glob("*.arrow"))
        for name in ("train", "validation", "test")
    }
    if any(not paths for paths in files.values()):
        raise FileNotFoundError(
            f"Missing PubChemQC JCIM2017 Arrow splits at {dataset}. Use the recovered "
            "data/pubchemqc/pubchemqc_jcim2017-split-v2 corpus; later PM6 datasets "
            "are not a substitute for its B3LYP/6-31G* charge labels."
        )
    required = set(
        DEFAULT_TOKEN_TARGETS
        + [
            "atomic-numbers",
            "atomic-coordinates",
            "bond-connections",
            "bond-order",
            "charge",
        ]
    )
    # Inspect schemas only: held-out labels are not consumed by preparation.
    for paths in files.values():
        split = load_dataset(
            "arrow", data_files=[str(p) for p in paths], split="train", streaming=True
        )
        if missing := required - set(split.column_names):
            raise ValueError(f"Not the expected JCIM2017 schema: missing {missing}")
    output = Path(output)
    link_directory(dataset, output)
    if provenance is None:
        return
    provenance = Path(provenance)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "dataset": os.path.relpath(output.absolute(), provenance.parent.absolute()),
                "release": "PubChemQC JCIM2017 B3LYP/6-31G*",
                "targets": DEFAULT_TOKEN_TARGETS,
                "source": "https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_2017.html",
                "splits": {
                    name: [str(p.relative_to(dataset)) for p in paths]
                    for name, paths in files.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory symlink to publish")
    parser.add_argument("--provenance", type=Path, help="Optional release and relative split-file metadata")
    args = parser.parse_args()
    materialize(args.dataset, args.output, args.provenance)


if __name__ == "__main__":
    main()
