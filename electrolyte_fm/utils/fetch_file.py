"""Download one pinned file using the system certificate store."""

import argparse
import hashlib
import json
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

import truststore


def verify(path: Path, size: int | None, digest: str | None) -> bool:
    if not path.is_file():
        return None
    if size is not None and path.stat().st_size != size:
        return False
    if digest is None:
        return True
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest() == digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=False, default=None)
    parser.add_argument("--bytes", type=int, required=False, default=None)
    parser.add_argument("--witness", type=Path, required=False)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not verify(args.output, args.bytes, args.sha256):
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as sink:
            temporary = Path(sink.name)
            try:
                context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                with urllib.request.urlopen(
                    args.url, context=context, timeout=60
                ) as source:
                    shutil.copyfileobj(source, sink)
                sink.flush()
                os.fsync(sink.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        if not verify(temporary, args.bytes, args.sha256):
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file does not match pinned size/hash: {args.url}"
            )
        temporary.replace(args.output)

    if args.witness is not None:
        args.witness.parent.mkdir(parents=True, exist_ok=True)
        args.witness.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "path": str(args.output.resolve()),
                    "sha256": args.sha256,
                    "bytes": args.bytes,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
