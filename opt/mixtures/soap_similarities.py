import json
import itertools

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize
from ase.data.pubchem import pubchem_atoms_search

solvent_molecules = {
    "O=C1OCC(F)O1": "FEC",
    "CC1COC(=O)O1": "PC",
    "CCOC(=O)OC": "EMC",
    "CCOC(=O)OCC": "DEC",
    "O=C1OCCO1": "EC",
    "COC(=O)OC": "DMC",
}


def calculate_similarity(smi1, smi2):
    """
    Calculate SOAP fingerprint and REMatch Kernel similarity
    """
    # Note: this only takes the first conformer, if there are multiple
    mol1 = pubchem_atoms_search(smiles=smi1)
    mol2 = pubchem_atoms_search(smiles=smi2)

    # Create SOAP descriptors
    # Parameters as in Kelly, C. (2024).
    # Excess Density as a Descriptor for Electrolyte Solvent Design (No. arXiv:2410.14689).
    soap = SOAP(
        species=["H", "C", "O", "F"],
        periodic=False,
        r_cut=10.0,
        n_max=15,
        l_max=15,
        rbf="gto",
        sigma=0.1,
    )
    descriptors1 = normalize(soap.create(mol1))
    descriptors2 = normalize(soap.create(mol2))

    # Calculate REMatch kernel
    kernel = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
    similarity = kernel.create([descriptors1, descriptors2])
    similarity = similarity[0, 1]
    return similarity


if __name__ == "__main__":
    results = {}
    for pair in list(itertools.combinations(list(solvent_molecules.keys()), 2)):
        results["_".join(pair)] = calculate_similarity(*pair)

    for smi in list(solvent_molecules.keys()):
        pair = (smi, smi)
        results["_".join(pair)] = calculate_similarity(*pair)

    with open("solvent_rematch.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
