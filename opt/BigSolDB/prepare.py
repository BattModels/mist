#!/usr/bin/env python
"""Build the BigSolDB finetuning target from the raw dataset."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

SMI_COLUMN = "SMILES"
SOLVENT_COLUMN = "Solvent"
TEMP_COLUMN = "T,K"
TARGET = "Solubility"
LOG_TARGET = "logS"

cli = typer.Typer()


def describe(df: pd.DataFrame, label: str):
    print(f"  {label:<26} {len(df):>6} rows  {df[SMI_COLUMN].nunique():>5} molecules")


@cli.command()
def main(
    input: Path = typer.Argument(..., help="raw BigSolDB csv"),
    output: Optional[Path] = None,
    solvent: str = "water",
    t_min: float = 293.0,
    t_max: float = 303.0,
    write_filtered: Optional[Path] = None,
):
    """Filter to one solvent and temperature window, keep one row per molecule,
    and add a log10 solubility column."""
    output = output or input.with_name("BigSolDB_water_room_temp_unique.csv")
    df = pd.read_csv(input)
    missing = {SMI_COLUMN, SOLVENT_COLUMN, TEMP_COLUMN, TARGET} - set(df.columns)
    assert not missing, f"{input} is missing columns: {sorted(missing)}"

    print(input)
    describe(df, "raw")

    solvents = set(df[SOLVENT_COLUMN].unique())
    assert solvent in solvents, (
        f"solvent {solvent!r} not in {input}; "
        f"{len(solvents)} available, e.g. {sorted(solvents)[:8]}"
    )
    df = df[df[SOLVENT_COLUMN] == solvent]
    describe(df, f"solvent == {solvent}")

    df = df[(df[TEMP_COLUMN] >= t_min) & (df[TEMP_COLUMN] <= t_max)]
    describe(df, f"{t_min:g} <= T <= {t_max:g} K")
    assert len(df), "temperature window selected no rows"

    if write_filtered:
        df.reset_index(drop=True).to_csv(write_filtered, index=False)
        print(f"  -> {write_filtered} (filtered, not deduplicated)")

    out = df.drop_duplicates(subset=SMI_COLUMN, keep="first").copy()
    describe(out, "dedup by SMILES (first)")
    assert out[SMI_COLUMN].is_unique, "SMILES still duplicated after dedup"

    assert (out[TARGET] > 0).all(), "log10 requires strictly positive solubility"
    out[LOG_TARGET] = np.log10(out[TARGET])
    out.to_csv(output, index=False)


if __name__ == "__main__":
    cli()
