#!/usr/bin/env python
"""Token counts for a SMILES CSV or a newline-delimited SMILES corpus."""

import collections
import json
import tarfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from electrolyte_fm.data_modules.utils import MolEncoding
from electrolyte_fm.utils.tokenizer import load_tokenizer

TOKENIZER = "mist-models/mist-28M-ti624ev1"
BATCH = 20_000

cli = typer.Typer()


def encode(smiles, encoding: str):
    """Apply the datamodule's MolEncoding, dropping molecules RDKit rejects."""
    enc = MolEncoding(encoding)
    if enc is MolEncoding.SMILES:
        return list(smiles), 0
    out = [enc(s) for s in smiles]
    return [s for s in out if s is not None], sum(s is None for s in out)


def as_table(counts, tok) -> pd.DataFrame:
    inverse = {v: k for k, v in tok.get_vocab().items()}
    return pd.DataFrame(
        [(inverse[i], n) for i, n in counts.most_common()], columns=["token", "count"]
    )


def write(table: pd.DataFrame, stats: dict, output: Path):
    table.to_csv(output, index=False)
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("wrote", output)


def iter_members(path: Path, split: str):
    """Yield binary file handles for data/{split}/*.txt, streaming the archive."""
    if path.is_dir():
        pattern = "**/*.txt" if split == "all" else f"**/{split}/*.txt"
        for f in sorted(path.glob(pattern)):
            with open(f, "rb") as fid:
                yield fid
        return

    with tarfile.open(path, "r|gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".txt")):
                continue
            if split != "all" and f"/{split}/" not in member.name:
                continue
            handle = tf.extractfile(member)
            if handle is not None:
                yield handle


@cli.command()
def csv(
    path: Path,
    smi_column: str = "SMILES",
    encoding: str = "smiles",
    tokenizer: str = TOKENIZER,
    output: Optional[Path] = None,
):
    """Count tokens in a CSV's SMILES column."""
    tok = load_tokenizer(tokenizer)
    df = pd.read_csv(path)
    assert smi_column in df.columns, f"{path} has no column {smi_column!r}"

    smiles, n_failed = encode(df[smi_column], encoding)
    encoded = tok(smiles)["input_ids"]
    counts = collections.Counter(i for seq in encoded for i in seq)
    table = as_table(counts, tok)

    stats = {
        "source": str(path),
        "encoding": encoding,
        "n_failed_encoding": n_failed,
        "n_molecules": len(smiles),
        "n_tokens": int(table["count"].sum()),
        "n_distinct": len(table),
        "vocab_size": len(tok),
        "n_unk": int(counts.get(tok.unk_token_id, 0)),
        "seq_mean": sum(map(len, encoded)) / len(encoded),
        "seq_max": max(map(len, encoded)),
        "tokenizer": tokenizer,
    }
    write(table, stats, output or path.with_name(f"{path.stem}_tokens.csv"))


@cli.command()
def corpus(
    path: Path,
    split: str = "train",
    stride: int = 1000,
    max_molecules: int = 0,
    encoding: str = "smiles",
    tokenizer: str = TOKENIZER,
    output: Optional[Path] = None,
):
    """Count tokens in a .tar.gz or directory of data/{split}/*.txt.

    The archive is streamed, never extracted. gzip is one sequential stream, so
    runtime is set by decompression; stride only reduces tokenization.
    """
    tok = load_tokenizer(tokenizer)
    counts: collections.Counter = collections.Counter()
    lengths: collections.Counter = collections.Counter()
    seen = sampled = files = n_failed = 0
    batch: list[str] = []
    t0 = time.time()

    def flush():
        nonlocal batch, n_failed
        if not batch:
            return
        smiles, failed = encode(batch, encoding)
        n_failed += failed
        for ids in tok(smiles)["input_ids"]:
            counts.update(ids)
            lengths[len(ids)] += 1
        batch = []

    capped = False
    for handle in iter_members(path, split):
        files += 1
        for raw in handle:
            seen += 1
            if seen % stride:
                continue
            smi = raw.decode("utf-8", "replace").strip()
            if not smi:
                continue
            batch.append(smi)
            sampled += 1
            if len(batch) >= BATCH:
                flush()
            if max_molecules and sampled >= max_molecules:
                capped = True
                break
        if files % 25 == 0:
            print(
                f"  {files:5d} files  {seen:>13,} lines  {sampled:>10,} sampled"
                f"  {(time.time() - t0) / 60:5.1f} min",
                flush=True,
            )
        if capped:
            break

    flush()
    table = as_table(counts, tok)
    stats = {
        "source": str(path),
        "split": split,
        "encoding": encoding,
        "stride": stride,
        "capped": capped,
        "n_failed_encoding": n_failed,
        "files": files,
        "lines_seen": seen,
        "molecules_sampled": sampled,
        "n_tokens": int(table["count"].sum()),
        "n_distinct": len(table),
        "vocab_size": len(tok),
        "n_unk": int(counts.get(tok.unk_token_id, 0)),
        "seq_mean": sum(k * v for k, v in lengths.items()) / max(sampled, 1),
        "seq_max": max(lengths) if lengths else 0,
        "tokenizer": tokenizer,
        "elapsed_min": (time.time() - t0) / 60,
    }
    stem = path.name.split(".")[0]
    write(table, stats, output or path.parent / f"{stem}_tokens.csv")


if __name__ == "__main__":
    cli()
