"""Discover published mixture splits without inventing held-out test data."""

from pathlib import Path


def split_files(configuration: Path) -> dict[str, list[str]]:
    """Require train/validation shards and allow an absent test directory."""
    files = {}
    for split in ("train", "validation", "test"):
        directory = configuration / split
        shards = sorted(directory.glob("*.arrow"))
        if not shards:
            if split != "test" or directory.exists():
                raise FileNotFoundError(
                    f"Missing mixture {split} Arrow shards: {directory}"
                )
            continue
        files[split] = [str(path) for path in shards]
    return files
