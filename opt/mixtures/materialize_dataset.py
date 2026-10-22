"""Stage the accepted Hugging Face mixture database and export its full table."""

from __future__ import annotations

import argparse
from pathlib import Path

from electrolyte_fm.utils.mixture_splits import split_files

REPOSITORY = "mist-models/excess-properties"
REVISION = "a42344b3057341a04ce6f2cf76f6f2882fda4e6c"
CONFIGURATIONS = ["random"] + [
    f"{method}-{fold}"
    for method in ("k-compound", "k-compound-strict")
    for fold in range(5)
]


def stage(output: Path, *, offline: bool = False) -> None:
    from huggingface_hub import snapshot_download

    from electrolyte_fm.utils.stage_hf import CACHE, link_model

    snapshot = Path(
        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=str(CACHE / "huggingface/hub"),
            local_files_only=offline,
        )
    ).resolve()
    if snapshot.name != REVISION:
        raise ValueError(f"Expected dataset revision {REVISION}; found {snapshot.name}")
    for configuration in CONFIGURATIONS:
        files = split_files(snapshot / configuration)
        if configuration == "random" and "test" not in files:
            raise FileNotFoundError(
                "The published random configuration requires its test split"
            )
    # The existing atomic directory-link helper applies equally to datasets.
    link_model(snapshot, output)


def export_table(source: Path, output: Path) -> int:
    """Use one complete configuration, preserving rows, nulls, and all columns."""
    from datasets import concatenate_datasets, load_from_disk

    random = source / "random"
    splits = split_files(random)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("The full database export requires all three random splits")
    datasets = [load_from_disk(str(random / split)) for split in splits]
    required = {
        "smi1",
        "smi2",
        "x1",
        "doi",
        "temperature [kelvin]",
        "pressure [megapascal]",
    }
    targets = [
        "density [gram / centimeter ** 3]",
        "molar volume [centimeter ** 3 / mole]",
        "molar enthalpy [joule / mole]",
    ]
    required.update(targets + ["excess " + name for name in targets])
    for dataset in datasets:
        if not required <= set(dataset.column_names):
            raise ValueError(
                f"Mixture database is missing columns: {sorted(required - set(dataset.column_names))}"
            )
        if not len(dataset):
            raise ValueError("The published random split must not be empty")
    table = concatenate_datasets(datasets)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(str(output), index=False)
    return len(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staged = commands.add_parser("stage")
    staged.add_argument("--output", type=Path, required=True)
    staged.add_argument("--offline", action="store_true")
    exported = commands.add_parser("export")
    exported.add_argument("--source", type=Path, required=True)
    exported.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        import truststore

        truststore.inject_into_ssl()
        stage(args.output, offline=args.offline)
    else:
        print(f"Exported {export_table(args.source, args.output)} mixture observations")


if __name__ == "__main__":
    main()
