import pubchempy
import typer
from rdkit import Chem

app = typer.Typer()


def get_smiles(name: str) -> str | None:
    results = pubchempy.get_compounds(name, namespace="name")
    if len(results) == 0:
        return None
    smi = Chem.MolToSmiles(Chem.MolFromInchi(results[0].inchi))
    return smi, results[0]


@app.command()
def smiles(name: str):
    if (out := get_smiles(name)) is None:
        exit(1)

    smi, record = out
    names = record.synonyms
    names = names[: (min(len(names), 5))]
    print("Other Names:", ", ".join(names))
    print("SMI:", smi)


if __name__ == "__main__":
    app()
