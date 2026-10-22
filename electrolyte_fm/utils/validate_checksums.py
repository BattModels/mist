"""Recursive output checksums with bounded state for Pomset's validator protocol."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def snapshot(paths):
    # Nested declared outputs are already covered by their parent directory.
    roots = sorted({Path(os.path.abspath(path)) for path in paths})
    roots = [
        path for path in roots if not any(parent in roots for parent in path.parents)
    ]
    checksum = hashlib.sha256()
    files = directories = total_bytes = 0

    def record(*values):
        checksum.update(json.dumps(values, separators=(",", ":")).encode("utf-8"))
        checksum.update(b"\0")

    def visit(path, relative):
        nonlocal files, directories, total_bytes
        before = path.lstat()
        if stat.S_ISDIR(before.st_mode):
            directories += 1
            record("directory", relative)
            for child in sorted(path.iterdir()):
                visit(child, relative + "/" + child.name)
        elif stat.S_ISLNK(before.st_mode):
            # Match validate_checksums(follow_symlinks=False), including cycles.
            record("symlink", relative, os.readlink(path))
        elif stat.S_ISREG(before.st_mode):
            files += 1
            record("file", relative, before.st_size)
            size = 0
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
                    size += len(chunk)
            after = path.stat()
            if (
                size != before.st_size
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ino != before.st_ino
            ):
                raise ValueError(f"Output changed while checksumming: {path}")
            total_bytes += size
        else:
            raise ValueError(f"Unsupported output file type: {path}")

    for root in roots:
        record("root", str(root))
        visit(root, ".")
    return {
        "validator": "recursive-sha256-v1",
        "sha256": checksum.hexdigest(),
        "files": files,
        "directories": directories,
        "bytes": total_bytes,
    }


def main():
    try:
        request = json.load(sys.stdin)
        state = snapshot(output["path"] for output in request["outputs"])
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 3
    prior = request.get("validation_state")
    if prior is not None and prior != state:
        print("Recursive output checksum state changed", file=sys.stderr)
        return 2
    print(json.dumps(state, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
