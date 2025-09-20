import logging

from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds, rdmolfiles, rdmolops


def robust_smiles_encode(mol: Mol, charge: int = 0):
    # Assign Stereochemistry from 3D
    if mol.GetNumConformers() > 0:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except IndexError:
            logging.debug("index error when finding bond orders for %s", mol)
        rdmolops.AssignStereochemistryFrom3D(mol)
    else:
        rdmolops.AssignStereochemistry(mol)

    # Cleanup molecule
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
    p.canonical = True
    p.allHsExplicit = False
    p.allBondsExplicit = False
    p.doKekule = False
    p.includeDativeBonds = False
    smi = Chem.MolToSmiles(mol, p)

    # Validate encoding
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise RuntimeError(
            "failed to convert %s to a valid SMILES: %s".format(mol, smi)
        )

    # Validate encoded molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        raise RuntimeError("failed to sanitize %s: %s".format(mol, smi))

    return smi
