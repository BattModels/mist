#!/usr/bin/env python
import concurrent.futures
import gzip
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdmolfiles, rdmolops

cli = typer.Typer()

logging.basicConfig(level=logging.INFO)


def extract_xyz(fid) -> Iterable[list[str]]:
    in_mol = True
    xyz_block = []
    for line in fid:
        if line := line.strip():
            xyz_block.append(line)
            in_mol = True
        elif in_mol:
            # Validate line count
            atoms = int(xyz_block[0])
            assert len(xyz_block) == atoms + 2
            yield "\n".join(xyz_block)
            xyz_block = []
            in_mol = False
        else:
            pass  # skip multiple empty lines


XYZ_META = re.compile(r"(?:(\w+) = (\w+))")


def xyz2mol(xyz_block: str) -> (str, int, dict):
    # Extract metadata
    meta = {}
    for m in XYZ_META.finditer(xyz_block.split("\n", maxsplit=2)[1]):
        meta[m.group(1)] = m.group(2)

    q = int(meta.get("q", 0))
    mol_id = meta["CSD_code"]
    for k in ["q", "S", "MND"]:
        if k in meta:
            meta[k] = int(meta[k])
    meta["xyz"] = xyz_block

    # Process Molecules
    logging.debug("processing %s with q: %d", mol_id, q)
    mol = Chem.MolFromXYZBlock(xyz_block)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=q)
    except IndexError:
        logging.debug("index error when finding bond orders for %s", mol_id)
    except ValueError as e:
        logging.error("failed to find bonds for %s: %s", mol_id, e)
        return None

    # Cleanup molecules
    rdmolops.AssignStereochemistryFrom3D(mol)
    mol = rdmolops.HapticBondsToDative(mol)
    rdmolops.SanitizeMol(mol, catchErrors=True)
    mol = rdmolops.RemoveHs(mol, sanitize=False)

    # # Round trip to smiles ( Get's rid of extra Hs)
    # # Yes, RemoveHs exists, but this (oddly) does a better job
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    rdmolops.SanitizeMol(mol, catchErrors=True)

    # Convert to SMILES
    p = rdmolfiles.SmilesWriteParams()
    p.isomericSmiles = True
    p.canonical = True
    p.allHsExplicit = False
    p.allBondsExplicit = False
    p.doKekule = False
    p.includeDativeBonds = False
    smi = Chem.MolToSmiles(mol, p)

    # Validate encoding
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logging.info("failed to convert %s to a valid SMILES: %s", mol_id, smi)
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        logging.info("failed to sanitize %s: %s", mol_id, smi)
        return None

    return mol_id, smi, meta


def cached_download(url: str, path: Path) -> Path:
    path = Path(path)
    cache = Path(__file__).parent.parent.parent.joinpath(".cache")
    cached_file = cache.joinpath(path)
    cached_file.parent.mkdir(exist_ok=True, parents=True)
    if not cached_file.exists():
        import urllib

        with urllib.request.urlopen(url) as fid:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_file, "wb") as out:
                out.write(fid.read())

    return cached_file


def get_xyz_block(tqmd, files):
    for file in files:
        with gzip.open(tqmd.joinpath(file), "rt") as fid:
            for xyz_block in extract_xyz(fid):
                yield xyz_block


@cli.command()
def build_dataset(
    base_url: str = "https://github.com/uiocompcat/tmQM/raw",
    commit: str = "bb1bc1c9f145a8ad2d35ce98ee5478fd01c5c48e",
    dataset_path=Path(__file__).parent.joinpath("data"),
    max_workers: int = 4,
):
    # Download the dataset
    for file in [
        "tmQM_X1.xyz.gz",
        "tmQM_X2.xyz.gz",
        "tmQM_X3.xyz.gz",
        "tmQM_X.q",
        "tmQM_y.csv",
    ]:
        url = f"{base_url}/{commit}/tmQM/{file}"
        p = cached_download(url, Path("tmqm", file))

    # Process XYZ files to SMILEs
    tqmd = p.parent
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        xyz_blocks = get_xyz_block(
            tqmd, ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz"]
        )
        records = executor.map(xyz2mol, xyz_blocks)

    # Unpack records, counting failures
    molecules = []
    failed_encode = 0
    for record in records:
        if record is None:
            failed_encode += 1
        else:
            mol_id, mol, meta = record
            molecules.append({"id": mol_id, "smiles": mol, **meta})

    # Join molecules with properties
    molecules = pd.DataFrame.from_records(molecules, index="id")
    properties = pd.read_csv(tqmd.joinpath("tmQM_y.csv"), index_col="CSD_code", sep=";")
    df = molecules.join(properties, on="id", how="left")

    logging.info(
        "failed to convert %d of %d xyz files to smiles",
        failed_encode,
        failed_encode + len(df.index),
    )

    # Save to disk
    ds = Dataset.from_pandas(df)
    ds_train_other = ds.train_test_split(test_size=0.2, seed=42)
    ds_val_test = ds_train_other["test"].train_test_split(test_size=0.5, seed=42)
    ds = DatasetDict(
        {
            "train": ds_train_other["train"],
            "validation": ds_val_test["train"],
            "test": ds_val_test["test"],
        }
    )
    ds.save_to_disk(dataset_path, num_shards={k: 8 for k in ds.keys()})


if __name__ == "__main__":
    cli()
