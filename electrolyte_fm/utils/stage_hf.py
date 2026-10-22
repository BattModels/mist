#!/usr/bin/env python
"""Stage a pinned Hugging Face snapshot as a model-directory symlink."""

from __future__ import annotations

# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import os
import tempfile
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / ".cache"

# Hugging Face reads auxiliary cache paths at import time.
if __name__ == "__main__":
    os.environ["HF_HOME"] = str(CACHE / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(CACHE)

from huggingface_hub import snapshot_download


def link_model(model: Path, output: Path, *, link_files: bool = False) -> None:
    """Publish a model link, optionally retaining a distinct output directory."""
    model = model.resolve(strict=True)
    if not model.is_dir():
        raise NotADirectoryError(model)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Resolve symlinked scratch/output ancestors, without following the link
    # being replaced. Relative targets are interpreted from the physical parent.
    output = output.parent.resolve() / output.name
    if link_files:
        sources = {source.name: source for source in model.iterdir()}
        if output.exists() and not output.is_symlink():
            if not output.is_dir() or any(
                not child.is_symlink() for child in output.iterdir()
            ):
                raise FileExistsError(f"refusing to replace an existing artifact: {output}")
            if {child.name for child in output.iterdir()} == sources.keys() and all(
                (output / name).resolve() == source.resolve()
                for name, source in sources.items()
            ):
                return
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            prepared = Path(temporary) / "model"
            prepared.mkdir()
            for name, source in sources.items():
                (prepared / name).symlink_to(os.path.relpath(source, output))
            previous = Path(temporary) / "previous"
            if output.exists() or output.is_symlink():
                output.rename(previous)
            try:
                prepared.replace(output)
            except BaseException:
                if previous.exists() or previous.is_symlink():
                    previous.rename(output)
                raise
        return
    if output.exists() and not output.is_symlink():
        raise FileExistsError(f"refusing to replace an existing artifact: {output}")
    if output.is_symlink() and output.resolve() == model:
        return
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        link = Path(temporary) / "model"
        # The link will live in output.parent after the atomic rename.
        link.symlink_to(os.path.relpath(model, output.parent), target_is_directory=True)
        os.replace(link, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Path of the model-directory symlink"
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--cache-ref",
        help="also register the pinned snapshot under this local cache ref",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        help="only stage matching repository files (repeatable)",
    )
    args = parser.parse_args()

    snapshot = Path(
        snapshot_download(
            args.model,
            revision=args.revision,
            cache_dir=str(CACHE / "huggingface/hub"),
            local_files_only=args.offline,
            allow_patterns=args.allow_pattern,
        )
    ).resolve()
    if snapshot.name != args.revision:
        raise ValueError(
            f"resolved revision {snapshot.name!r}, expected {args.revision!r}"
        )
    if not snapshot.is_dir() or not any(snapshot.iterdir()):
        raise ValueError(f"snapshot contains no files: {snapshot}")
    if args.cache_ref:
        ref = Path(args.cache_ref)
        if ref.is_absolute() or ".." in ref.parts:
            raise ValueError("--cache-ref must be a relative cache ref")
        ref_path = snapshot.parents[1] / "refs" / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        # Hugging Face cache refs contain the raw commit id with no newline.
        ref_path.write_text(args.revision)
    link_model(snapshot, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
