#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "rdkit",
#     "typer",
# ]
# ///

# python
import re
import csv
from pathlib import Path
import pandas as pd
from rdkit import Chem
import typer

AMBIGUOUS_PAIRS = ["Sn", "Sb", "Co", "Os", "Po", "Sc", "Cs", "Cn"]

app = typer.Typer()


def canonicalize_smiles(smiles: str):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def count_ambiguous_substrings(smiles: str):
    """Count occurrences of each ambiguous substring in a SMILES string."""
    smiles = re.sub(r"\[[^\]]+]", "", smiles)  # Ignore bracketed atoms
    return {pair: len(re.findall(pair, smiles)) for pair in AMBIGUOUS_PAIRS}


def get_chunks(input_file, chunk_size):
    """Read CSV file dynamically handling .csv and .csv.gz extensions in chunks."""
    if input_file.suffix == ".gz":
        return pd.read_csv(input_file, compression="gzip", chunksize=chunk_size)
    return pd.read_csv(input_file, chunksize=chunk_size)


def write_header(output_file, ambiguity_keys):
    """Write the header to the output CSV file."""
    header = ["name", "cid", "smiles", "total_ambiguity_count"] + ambiguity_keys
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_to_csv(output_file, row):
    """Append a row to the output CSV file."""
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def process_chunk(chunk, output_file):
    """Process a single chunk and append results incrementally to the output file."""
    for _, row in chunk.iterrows():
        original = row["SMILES"]
        canonical = canonicalize_smiles(original)
        if canonical:
            ambiguity_counts = count_ambiguous_substrings(canonical)
            total_count = sum(ambiguity_counts.values())
            if total_count > 0:
                output_row = {
                    "name": row["Name"],  # Include Name column
                    "cid": row["Compound_CID"],  # Include Compound_CID column
                    "smiles": canonical,  # Canonical SMILES column
                    "total_ambiguity_count": total_count,  # Total count column
                }
                output_row.update(ambiguity_counts)  # Add individual counts
                append_to_csv(output_file, output_row)


@app.command("filter")
def process_smiles(
    input_file: str,
    chunk_size: int = typer.Option(10000, help="Number of rows to process per chunk."),
):
    """
    Process SMILES strings in the input file to identify ambiguities and write results incrementally.
    """
    input_path = Path(input_file)
    output_file = input_path.with_name(input_path.stem + "_ambiguous.csv")

    # Initialize output file with header
    ambiguity_keys = list(
        count_ambiguous_substrings("").keys()
    )  # Keys for ambiguous pairs
    header = [
        "name",
        "cid",
        "smiles",
        "InChI",
        "IUPAC_Name",
        "total_ambiguity_count",
    ] + ambiguity_keys
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    total_molecules_checked = 0  # Counter for molecules
    chunks = get_chunks(input_path, chunk_size)
    for chunk in chunks:
        if not {"SMILES", "Name", "Compound_CID", "InChI", "IUPAC_Name"}.issubset(
            chunk.columns
        ):
            typer.echo(
                "CSV must contain 'SMILES', 'Name', 'Compound_CID', 'InChI', and 'IUPAC_Name' columns.",
                err=True,
            )
            raise typer.Exit()

        for _, row in chunk.iterrows():
            total_molecules_checked += 1  # Increment counter
            original = row["SMILES"]
            canonical = canonicalize_smiles(original)
            if canonical:
                ambiguity_counts = count_ambiguous_substrings(canonical)
                total_count = sum(ambiguity_counts.values())
                if total_count > 0:
                    output_row = {
                        "name": row["Name"],  # Include Name column
                        "cid": row["Compound_CID"],  # Include Compound_CID column
                        "smiles": canonical,  # Canonical SMILES column
                        "InChI": row["InChI"],  # InChI source column
                        "IUPAC_Name": row["IUPAC_Name"],  # IUPAC Name column
                        "total_ambiguity_count": total_count,  # Total count column
                    }
                    output_row.update(ambiguity_counts)  # Add individual counts
                    with open(output_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=output_row.keys())
                        writer.writerow(output_row)

    # Print total molecules checked to stderr
    typer.echo(f"Total molecules checked: {total_molecules_checked}", err=True)
    typer.echo(f"Results written incrementally to: {output_file}")


@app.command("stats")
def tabulate_ambiguities(
    output_file: str,
    remove_nonbond: bool = True,
    ignore_cn: bool = False,
):
    """
    Tabulate the total counts of each ambiguity type and display shortest 5 examples of each.
    Optionally, remove SMILES strings containing a period ("."), and ignore "Cn" in the distinct ambiguities calculation.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        typer.echo(f"File {output_file} does not exist.")
        raise typer.Exit()

    # Read the output file
    df = pd.read_csv(output_file)

    # Deduplicate on the `smiles` column
    df = df.drop_duplicates(subset=["smiles"])

    # Optionally remove rows where `smiles` contains a period (".")
    if remove_nonbond:
        df = df[~df["smiles"].str.contains(r"\.", na=False)]
        typer.echo("Removed SMILES containing a nonbond ('.').")

    # Ensure required ambiguity columns exist
    ambiguity_keys = [key for key in AMBIGUOUS_PAIRS if key in df.columns]

    if not ambiguity_keys:
        typer.echo("No ambiguity type columns found in the file.")
        raise typer.Exit()

    if ignore_cn and "Cn" in ambiguity_keys:
        ambiguity_keys.remove("Cn")
        typer.echo("Ignoring 'Cn' in distinct ambiguities calculation.")

    # Calculate totals for each ambiguity type
    ambiguity_totals = df[ambiguity_keys].sum().sort_values(ascending=False)

    # Display results in tabular form
    typer.echo("\nTabulated Ambiguities:")
    typer.echo("-" * 40)
    for ambiguity, count in ambiguity_totals.items():
        typer.echo(f"{ambiguity}: {count}")

        # Extract shortest 5 SMILES examples for this ambiguity type
        filtered_df = df[df[ambiguity] > 0]
        shortest_examples = filtered_df.assign(
            smiles_length=filtered_df["smiles"].str.len()
        ).nsmallest(5, "smiles_length")
        typer.echo(f"Shortest 5 examples for {ambiguity}:")
        for _, row in shortest_examples.iterrows():
            typer.echo(f"\t{row['name']}: {row['smiles']}")
        typer.echo("-" * 40)

    # Calculate the number of distinct ambiguity types per molecule
    df["distinct_ambiguities"] = df[ambiguity_keys].gt(0).sum(axis=1)

    # Sort by number of distinct ambiguities and then by SMILES length
    df = df.assign(smiles_length=df["smiles"].str.len())
    top_distinct = df.sort_values(
        by=["distinct_ambiguities", "smiles_length"], ascending=[False, True]
    ).head(5)

    # Print top 5 molecules with the greatest number of distinct ambiguities
    typer.echo("\nTop 5 molecules with the greatest number of distinct ambiguities:")
    typer.echo("-" * 40)
    for _, row in top_distinct.iterrows():
        typer.echo(
            f"{row['name']} (CID: {row['cid']}): {row['smiles']} - {row['distinct_ambiguities']} distinct ambiguities"
        )
    typer.echo("-" * 40)


@app.command("merge")
def merge_ambiguous_files(
    output_file: str,
    input_files: list[str] = typer.Argument(
        ..., help="List of ambiguous.csv files to merge."
    ),
):
    """
    Merge multiple ambiguous.csv files and deduplicate using InChIKey.
    """
    combined_df = pd.DataFrame()
    for file in input_files:
        input_path = Path(file)
        if not input_path.exists():
            typer.echo(f"File {file} does not exist.")
            raise typer.Exit()

        typer.echo(f"Reading file: {file}")
        df = pd.read_csv(file)

        if "InChI" not in df or "smiles" not in df:
            typer.echo(f"File {file} must contain 'InChI' and 'smiles' columns.")
            raise typer.Exit()

        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Deduplicate using InChIKey
    if "InChI" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["InChI"])

    typer.echo(f"Merged and deduplicated {len(combined_df)} rows based on InChIKey.")

    # Write to output file
    output_path = Path(output_file)
    combined_df.to_csv(output_path, index=False)
    typer.echo(f"Final merged file written to: {output_path}")


if __name__ == "__main__":
    app()
