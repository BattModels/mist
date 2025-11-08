import csv
import sqlite3
import subprocess
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from itertools import islice

import typer

cli = typer.Typer()

logging.basicConfig(level=logging.INFO)


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


@cli.command()
def fragment_smi(
    file,
    frag_file: str | None = None,
    w: int = 150,
    smi_column: str = "smi",
    name_column: str | None = None,
):
    # Delay import to reduce startup time
    frag_file = frag_file or "fragments.smi"
    with NamedTemporaryFile() as outfile:
        with (
            open(file, "r", newline="") as infile,
            open(outfile.name, "w", newline="") as outfile,
        ):
            name_column = name_column or "name"
            fieldnames = (
                None if str(file).endswith(".csv") else [smi_column, name_column]
            )
            delimiter = "," if str(file).endswith(".csv") else "\t"
            reader = csv.DictReader(infile, fieldnames, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter="\t")
            # Write filtered rows
            for row in reader:
                writer.writerow(
                    [row[smi_column], row.get(name_column, row[smi_column])]
                )

        subprocess.run(
            [
                Path(__file__).parent.parent.joinpath(
                    "vendor", "FASMIFRA", "bin", "fasmifra_fragment.py"
                ),
                "-i",
                outfile.name,
                "-w",
                str(w),
                "-o",
                str(frag_file),
            ]
        )
    return frag_file


@cli.command("store-fragments")
def store_fragments_in_sqlite(folder):
    db_path = Path(folder, "fragments.sqlite")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    def extract_fragments(folder_path):
        folder = Path(folder_path)
        for file_path in folder.glob("*.frag"):
            logging.info("processing %s", file_path)
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        yield (line.strip().split()[0],)

    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS fragments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fragment TEXT UNIQUE
        )
    """)

    # Insert patterns, ignoring duplicates
    for batch in batched(extract_fragments(folder), 100_000):
        c.executemany(
            "INSERT OR IGNORE INTO fragments (fragment) VALUES (?)",
            batch,
        )
        conn.commit()

    conn.close()


if __name__ == "__main__":
    cli()
