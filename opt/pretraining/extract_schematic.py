"""Extract the unchanged explanatory schematic from the pinned paper source."""

import argparse
import hashlib
import tarfile
from pathlib import Path

SOURCE_SHA256 = "82e4af73525de5f4a196e36da205edee3e0977c0b1ffb734812247b6b7c014c7"


def extract(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Supply the original paper source archive at {source}")
    with source.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError("Paper source archive does not match the pinned v3 release")
    with tarfile.open(source) as archive:
        member = archive.getmember("scaling_workflow.pdf")
        if not member.isfile():
            raise ValueError("Expected a regular scaling_workflow.pdf asset")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Unreadable scaling_workflow.pdf asset")
        payload = stream.read()
    if not payload.startswith(b"%PDF-"):
        raise ValueError("The released schematic is not a PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)
