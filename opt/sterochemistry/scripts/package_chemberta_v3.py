"""Bundle ChemBERTa-v3's pinned MolFormer code into its model artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

CODE_REPOSITORY = "ibm/MoLFormer-XL-both-10pct"


def local_reference(value):
    if isinstance(value, list):
        return [local_reference(item) for item in value]
    if isinstance(value, str) and "--" in value:
        repository, reference = value.split("--", 1)
        if repository != CODE_REPOSITORY:
            raise ValueError(f"Unstaged custom-code repository: {repository}")
        return reference
    return value


def package(model: Path, code: Path, output: Path) -> None:
    model, code = model.resolve(strict=True), code.resolve(strict=True)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    if any(
        output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve())
        for source in (model, code)
    ):
        raise ValueError("Output must be separate from the staged inputs")
    metadata = {}
    for name in ("config.json", "tokenizer_config.json"):
        path = model / name
        if not path.is_file():
            if name == "config.json":
                raise FileNotFoundError(path)
            continue
        document = json.loads(path.read_text())
        if "auto_map" in document:
            document["auto_map"] = {
                key: local_reference(value) for key, value in document["auto_map"].items()
            }
        metadata[name] = document
    sources = sorted(code.rglob("*.py"))
    if not sources:
        raise ValueError(f"No custom Python code in {code}")
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        prepared = Path(temporary) / "model"
        prepared.mkdir()
        for source in model.rglob("*"):
            if source.is_file():
                target = prepared / source.relative_to(model)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Interpret links from their final location after publication.
                final_parent = output / target.parent.relative_to(prepared)
                target.symlink_to(os.path.relpath(source, final_parent))
        for source in sources:
            target = prepared / source.relative_to(code)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            shutil.copyfile(source, target)
        for name, document in metadata.items():
            target = prepared / name
            target.unlink(missing_ok=True)
            target.write_text(json.dumps(document, indent=2) + "\n")
        if output.is_symlink():
            raise FileExistsError(f"Refusing to replace a directory symlink: {output}")
        if output.exists():
            shutil.rmtree(output)
        prepared.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "code", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    package(args.model, args.code, args.output)


if __name__ == "__main__":
    main()
