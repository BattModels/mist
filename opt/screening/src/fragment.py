import sys
import csv
import subprocess
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from electrolyte_fm.data_modules.utils import MolEncoding


def fragment_smi(file, frag_file: str | None = None):
    frag_file = frag_file or "fragments.smi"
    encoder = MolEncoding.KEKULE
    with NamedTemporaryFile() as outfile:
        with (
            open(file, "r", newline="") as infile,
            open(outfile.name, "w", newline="") as outfile,
        ):
            fieldnames = None if str(file).endswith(".csv") else ["smi", "name"]
            reader = csv.DictReader(infile, fieldnames)
            writer = csv.writer(outfile, delimiter="\t")
            # Write filtered rows
            for row in reader:
                writer.writerow([encoder(row["smi"]), row["name"] or row["smi"]])

        subprocess.run(
            [
                Path(__file__).parent.parent.joinpath(
                    "vendor", "FASMIFRA", "bin", "fasmifra_fragment.py"
                ),
                "-i",
                outfile.name,
                "-w",
                "6",
                "-o",
                str(frag_file),
            ]
        )
    return frag_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("smi", type=str, help="SMILES file to fragment")
    p.add_argument("frag_file", type=str, help="Fragment file to write")
    args = p.parse_args()
    fragment_smi(args.smi, args.frag_file)
