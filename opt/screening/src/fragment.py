import csv
import logging
import sqlite3
import subprocess
import sys
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import typer

cli = typer.Typer()
logger = logging.getLogger(__name__)

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
    smi_column: str = "smi",
    name_column: str | None = None,
    num_passes: int = 1,
    fragmenter: Path | None = None,
    interpreter: str = sys.executable,
):
    # Delay import to reduce startup time
    frag_file = frag_file or "fragments.smi"
    Path(frag_file).parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile() as outfile:
        with (
            open(file, "r", newline="") as infile,
            open(outfile.name, "w", newline="") as outfile,
        ):
            name_column = name_column or "name"
            if str(file).endswith(".csv"):
                reader = csv.DictReader(infile)
            else:
                reader = (
                    {
                        smi_column: fields[0],
                        name_column: fields[1] if len(fields) > 1 else fields[0],
                    }
                    for line in infile
                    if (fields := line.strip().split(maxsplit=1))
                )
            writer = csv.writer(outfile, delimiter="\t")
            # Write filtered rows
            for row in reader:
                writer.writerow(
                    [row[smi_column], row.get(name_column) or row[smi_column]]
                )

        subprocess.run(
            [
                interpreter,
                fragmenter,
                "-i",
                outfile.name,
                "-o",
                str(frag_file),
                "-n",
                str(num_passes),
            ],
            check=True,
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
            logger.info("processing %s", file_path)
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        yield (line.strip().split()[0],)

    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS fragments (
            fragment TEXT PRIMARY KEY
        ) WITHOUT ROWID
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
