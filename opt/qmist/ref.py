#!/usr/bin/env -S uv run python
import csv
import json
import logging
import sys
from urllib.request import urlopen
from pathlib import Path
from typing import IO

import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

COLUMNS = [
    "smiles",
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
]

cli = typer.Typer()


def download_csv(url: str) -> IO[bytes]:
    return urlopen(url)


def parse_csv(file_obj: IO[bytes], columns: list[str]) -> list[dict[str, float]]:
    decoded_lines = (line.decode("utf-8") for line in file_obj)
    reader = csv.DictReader(decoded_lines)
    result = []
    for row in reader:
        parsed_row = {}
        for col in columns:
            value = row[col]
            if col == "smiles":
                parsed_row[col] = value
            else:
                parsed_row[col] = float(value)
        result.append(parsed_row)
    return result


def write_jsonl(data: list[dict[str, float]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


@cli.command("generate")
def generate(output: str = "qm9.jsonl"):
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    csv_stream = download_csv(url)
    data = parse_csv(csv_stream, COLUMNS)
    write_jsonl(data, output)


@cli.command("compare")
def compare(
    input: typer.FileText = typer.Option(
        sys.stdin, help="Input JSON object file or stdin."
    ),
    ref: Path = typer.Option("qm9.jsonl", help="Reference JSONL file."),
):
    query = json.load(input)
    smiles = str(query.get("smiles"))
    if not smiles:
        logging.exception("Missing 'smiles' key in input.")

    with ref.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("smiles") == smiles:
                break
        else:
            logging.exception(f"SMILES '{smiles}' not found in {ref}")

    for key in COLUMNS:
        if key == "smiles":
            continue
        val_query = query.get(key)
        val_ref = entry.get(key)

        if val_query is None or val_ref is None:
            logging.warning(f"Skipping {key}: missing in input or reference")
            continue

        diff = val_query - val_ref
        rel_diff = diff / (abs(val_ref) + 1e-6)

        print(
            f"{key:<12} Δ = {diff:.6g} ({rel_diff:.3%}) (input={val_query:.6g}, ref={val_ref:.6g})"
        )


if __name__ == "__main__":
    cli()
