import traceback
import math
import json
import logging
import tarfile
import warnings
from pathlib import Path
from typing import Optional

import typer
import torch
from ase import Atom, Atoms
from ase.thermochemistry import IdealGasThermo
from datasets import Dataset, DatasetDict, load_dataset
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from tqdm import tqdm

cli = typer.Typer()

logging.basicConfig(level=logging.INFO)

HARTREE_TO_EV = 27.211_386_245_981
EV_TO_HARTREE = 1 / HARTREE_TO_EV


@cli.command()
def ls(path: str):
    with tarfile.open(path, "r:gz") as archive:
        for file in archive.getmembers():
            print(f"{file.path}: {file.size} bytes")


@cli.command()
def extract(path: str, output: Optional[str] = None):
    output = output or str(Path(".", Path(path).name))
    with tarfile.open(path, "r:gz") as archive:
        for file in tqdm(archive.getmembers()):
            if not file.path.endswith(".tar.gz"):
                continue

            compounds = extract_compounds(archive, file)
            write_compounds(compounds, output, file)


@cli.command()
def package(path: str, output: Optional[str] = None, num_shards: Optional[int] = None):
    output = output or path

    paths = list(Path(path).glob("*.jsonl"))
    ds = Dataset.from_generator(
        iterate_compounds,
        gen_kwargs={"paths": paths},
        keep_in_memory=False,
        num_proc=64,
    )
    ds.save_to_disk(output, max_shard_size="1GB")


@cli.command()
def split(path: str, output: str):
    ds = load_dataset(
        "arrow",
        data_files=[str(Path(path, "*.arrow"))],
        keep_in_memory=False,
        split="train",
    )

    def random_split(batch: dict[str, list]):
        n = len(next(iter(batch.values())))
        batch["__split_key"] = torch.rand(n).tolist()
        return batch

    def filter_split(split_key: list[float], lower: float = 0, upper: float = 1):
        split_key = torch.tensor(split_key)
        f = (lower <= split_key) & (split_key < upper)
        return f.tolist()

    # Split dataset by randomly assigning a split key
    ds = ds.map(random_split, batched=True)
    ds_train = ds.filter(
        filter_split, input_columns="__split_key", fn_kwargs={"lower": 0, "upper": 0.8}
    ).remove_columns("__split_key")
    ds_val = ds.filter(
        filter_split,
        input_columns="__split_key",
        fn_kwargs={"lower": 0.8, "upper": 0.9},
    ).remove_columns("__split_key")
    ds_test = ds.filter(
        filter_split, input_columns="__split_key", fn_kwargs={"lower": 0.9, "upper": 1}
    ).remove_columns("__split_key")

    # Split the dataset
    ds = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )
    ds.save_to_disk(output, max_shard_size="1GB")


def iterate_compounds(paths):
    for path in paths:
        with open(path, "r") as fid:
            for line in fid:
                try:
                    yield normalize_compound(json.loads(line))
                except Exception:
                    logging.error("Skipping %s due to %s", line, traceback.format_exc())


def normalize_compound(compound: dict):
    out = {}
    for key in ["smiles", "inchi", "inchikey"]:
        out[key] = compound[key]

    properties = compound["properties"]
    out["partial-charge-mulliken"] = [
        float(c) for c in properties["partial charges"]["mulliken"]
    ]
    out["partial-charge-lowdin"] = [
        float(c) for c in properties["partial charges"]["lowdin"]
    ]
    out["dipole-moment"] = float(properties["total dipole moment"])
    for etype in ["alpha", "beta"]:
        for k, v in properties["energy"][etype].items():
            out[etype + "-" + k] = float(v)

    out["total-energy"] = float(properties["energy"]["total"])
    out["charge"] = float(properties["charge"])

    # Structural information
    out["atomic-numbers"] = [int(n) for n in compound["atoms"]["elements"]["number"]]
    out["atomic-coordinates"] = [float(p) for p in compound["atoms"]["coords"]["3d"]]
    if len(out["atomic-numbers"]) == 1:
        out.update({"bond-connections": list(), "bond-order": list()})
    else:
        out["bond-connections"] = [
            int(i) for i in compound["bonds"]["connections"]["index"]
        ]
        out["bond-order"] = [int(i) for i in compound["bonds"]["order"]]

    assert len(out["atomic-numbers"]) == properties["number of atoms"]
    assert len(out["atomic-coordinates"]) == 3 * properties["number of atoms"]
    assert 2 * len(out["bond-order"]) == len(out["bond-connections"])

    return out


def compute_thermo(
    atomic_number: list[int],
    positions: list[float],
    orbital_energies: list[float],
    total_energy: float,
    smiles: str,
    temperature: float = 298.15,
    pressure: float | None = None,
) -> tuple[float, float, float]:
    """

    # Parameters
    atomic_number: Atomic number for each atom
    positions: Flatten list of position in Å [x1, y1, z1, x2, y2, z2, ...]
    orbital_energies: Orbital energies in eV
    total_energy: Total energy in eV
    smiles: SMILES string for the molecules
    temperature: Temperature in Kelvin to compute thermodynamics at
    pressure: Pressure in bar to compute thermodynamics at, defaults to the ref pressure

    # Returns
    h: Enthalpy in hartrees
    gibbs: Gibbs energy in hartrees
    zpve: Zero-point vibrational energy in hartrees

    """
    assert len(atomic_number) * 3 == len(positions)
    molecule = Atoms(
        [
            Atom(atomic_number, positions[3 * idx : 3 * idx + 3])  # type: ignore (atomic_number can be an int)
            for idx, atomic_number in enumerate(atomic_number)
        ]
    )
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    radicals = NumRadicalElectrons(mol)

    # Pick geometry based on number of atoms
    num_atoms = len(atomic_number)
    if num_atoms == 1:
        geometry = "monatomic"
    elif num_atoms == 2:
        geometry = "linear"
    else:
        geometry = "nonlinear"

    # Suppress warnings about imaginary energies (Filtering them explicitly)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^\d+ imag modes removed$",
            module="ase.thermochemistry",
        )

        thermo = IdealGasThermo(
            vib_energies=orbital_energies,
            potentialenergy=total_energy,
            geometry=geometry,
            atoms=molecule,
            spin=radicals / 2,  # Ignore electronic entropy
            symmetrynumber=len(mol.GetSubstructMatches(mol, uniquify=False)),
            ignore_imag_modes=True,
        )
        h = EV_TO_HARTREE * thermo.get_enthalpy(temperature, verbose=True)
        assert not math.isinf(h) and not math.isnan(h)
        gibbs = EV_TO_HARTREE * thermo.get_gibbs_energy(
            temperature,
            pressure or thermo.referencepressure,
            verbose=False,
        )
        assert not math.isinf(gibbs) and not math.isnan(gibbs)
        zpve = EV_TO_HARTREE * thermo.get_ZPE_correction()
        assert not math.isinf(zpve) and not math.isnan(zpve)

    return h, gibbs, zpve


def write_compounds(iter, output: str, file: tarfile.TarInfo):
    output_file = Path(output, Path(file.path).name.replace(".tar.gz", ".jsonl"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fid:
        for row in iter:
            json.dump(row, fid)
            fid.write("\n")


def extract_compounds(archive: tarfile.TarFile, file: tarfile.TarInfo):
    with tarfile.open(
        fileobj=archive.extractfile(file), mode="r:gz"
    ) as compounds_archive:
        for compound in compounds_archive.getmembers():
            if not (compound.isfile() and compound.path.endswith(".json")):
                continue

            data = compounds_archive.extractfile(compound)
            if data is None:
                logging.warning("empty compound %s in %s", compound.path, file.path)
                continue

            yield json.load(data)


if __name__ == "__main__":
    cli()
