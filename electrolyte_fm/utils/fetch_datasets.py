#!/usr/bin/env python
"""Download a pinned dataset archive and safely materialize its contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import truststore


CHUNK_BYTES = 8 * 1024 * 1024


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(path: Path, expected_bytes: int, expected_md5: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"unexpected archive size: {path.stat().st_size}")
    actual_md5 = digest(path, "md5")
    if actual_md5 != expected_md5.lower():
        raise ValueError(f"unexpected archive MD5: {actual_md5}")


def fetch(url: str, destination: Path, expected_bytes: int, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            verify(destination, expected_bytes, expected_md5)
            return
        except ValueError:
            pass
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            with urllib.request.urlopen(url, context=context) as response:  # noqa: S310
                shutil.copyfileobj(response, output, length=CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        verify(temporary, expected_bytes, expected_md5)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stripped_path(name: str, components: int) -> PurePosixPath | None:
    source = PurePosixPath(name)
    if not name or source.is_absolute() or ".." in source.parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    # Preserve a leading ``./`` as a component, matching GNU tar's
    # --strip-components behavior used by the paper-release archive.
    parts = tuple(part for part in name.split("/")[components:] if part)
    return PurePosixPath(*parts) if parts else None


def extract(
    archive_path: Path, destination: Path, strip_components: int, required: list[str]
) -> None:
    if destination.exists():
        missing = [name for name in required if not (destination / name).exists()]
        if not missing:
            return
        raise ValueError(f"incomplete existing dataset at {destination}: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                relative = stripped_path(member.name, strip_components)
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"archive links are not permitted: {member.name!r}"
                    )
                if not (member.isdir() or member.isfile()):
                    raise ValueError(
                        f"archive special file is not permitted: {member.name!r}"
                    )
                if relative is None:
                    continue
                target = temporary.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_BYTES)
        missing = [name for name in required if not (temporary / name).exists()]
        if missing:
            raise ValueError(f"archive is missing required dataset paths: {missing}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--strip-components", type=int, default=0)
    parser.add_argument("--required", action="append", default=[])
    args = parser.parse_args()

    fetch(args.url, args.archive, args.bytes, args.md5)
    extract(args.archive, args.extract, args.strip_components, args.required)
    receipt = {
        "url": args.url,
        "archive": str(args.archive.resolve()),
        "bytes": args.bytes,
        "md5": args.md5.lower(),
        "extract": str(args.extract.resolve()),
    }
    output = args.extract / "fetch-receipt.json"
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
