#!/usr/bin/env -S uv run python

import json
from itertools import product

import typer
from generate import ARENE_SMILES, generate_metal_arene_complex

from electrolyte_fm.utils.encode import robust_smiles_encode

cli = typer.Typer()


def generate_molecules(binding: str = "auto"):
    organics = ARENE_SMILES.keys()
    metals = ["Ti", "Fe", "Ni", "Cr"]
    charges = [0, 1, 2]
    for organic, metal, charge in product(organics, metals, charges):
        if organic in ["pyrene", "coronene"]:
            config = ["inner", "outer"]
        else:
            config = ["auto"]
        for conf in config:
            yield {
                "smiles": ARENE_SMILES[organic],
                "metal": "organic",
                "organic": organic,
                "placement": "none",
                "charge": -1,
            }

            smi = robust_smiles_encode(
                generate_metal_arene_complex(
                    organic,
                    metal,
                    charge=charge,
                    configuration=conf,
                    binding=binding,
                )
            )
            yield {
                "smiles": smi,
                "metal": metal,
                "organic": organic,
                "charge": charge,
                "placement": conf,
            }


def flatten_dict(d: dict[str, dict | list]):
    keys = list(d.keys())
    cols = [d[k]["value"] for k in keys]
    for row in zip(*cols, strict=True):
        yield {k: float(x) for k, x in zip(keys, row)}


@cli.command()
def molecules():
    for smi in generate_molecules():
        print(smi)


@cli.command()
def run(save_directory: str, binding: str = "auto"):
    from electrolyte_fm.models.prod_finetune import MISTFinetuned

    # Load model
    model = MISTFinetuned.from_pretrained(save_directory)
    model = model.eval().to("mps")

    # Generate and predict
    molecules = list(generate_molecules(binding=binding))
    smiles = [x["smiles"] for x in molecules]
    results = model.predict(smiles)

    # Flatten and export results as json
    results = flatten_dict(results)
    out = []
    for r, mol in zip(results, molecules):
        r.update(mol)
        out.append(r)
    print(json.dumps(out))


if __name__ == "__main__":
    cli()
