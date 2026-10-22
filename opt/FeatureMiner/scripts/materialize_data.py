"""Fetch the released Lipinski training/validation data used by the paper."""

import argparse
import json
from pathlib import Path

from electrolyte_fm.utils.fetch_datasets import extract, fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    archive = root.parent / "datasets.tar.xz"
    fetch(
        "https://zenodo.org/api/records/21302713/files/datasets.tar.xz/content",
        archive,
        685944,
        "ec2871b2f78986ac3df6c85e38a3acec",
    )
    source = root.parent / "dataset-archives"
    extract(archive, source, 1, ["lipinski.tar.gz"])
    extract(source / "lipinski.tar.gz", root, 1, ["data/train", "data/validation"])
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "source": "zenodo:21302713/datasets.tar.xz:datasets/lipinski.tar.gz",
                "splits": ["train", "validation"],
                "test_used": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
