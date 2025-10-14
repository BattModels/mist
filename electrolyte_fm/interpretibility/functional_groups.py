from rdkit import Chem

# SMARTS patterns for each functional group
functional_group_smarts = {
    "alkane": "[CH3][CH3]",  # At least 2 member
    "alkene": "C=C",
    "alkyne": "C#C",
    "arene": "a",
    "alcohol": "[CX4][OH]",
    "ether": "[OD2]([#6])[#6]",
    "carbonate ester": "[#6][OX2H0]C(=O)[OX2H0][#6]",
    "epoxide": "[OD2]1[CH2][CH2]1",
    "haloalkane": "[CX4][F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic acid": "C(=O)[OH]",
    "acid anhydride": "C(=O)OC(=O)",
    "ester": "[#6][CX3](=O)[OX2H0][#6;!$(C(=O))]",
    "amide": "[#6][CX3](=O)[NX3;H2,H1,H0]",
    "acyl halide": "C(=O)[F,Cl,Br,I]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "C#N",
    "imine": "[NX2]=[CX3]",
    "isocyanate": "N=C=O",
    "azo compound": "N=N",
    "thiol": "[SH]",
    "aqueous": "[OH2]",
    "salt": "[+1,+2,+3].[-1,-2,-3]",
}

# Compile SMARTS into RDKit molecules
functional_group_mols = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in functional_group_smarts.items()
}


def identify_functional_groups(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    found = []

    for name, pattern in functional_group_mols.items():
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            continue
        if name == "dinitrile":
            if len(matches) >= 2:
                found.append("dinitrile")
        else:
            found.append(name)

    # Post-process matches
    if "acid anhydride" in found:
        found = [f for f in found if f not in {"ester", "ether"}]
    if "carbonate ester" in found:
        found = [f for f in found if f not in {"ester", "ether"}]
    if "epoxide" in found:
        found = [f for f in found if f != "ether"]
    if "ester" in found:
        found = [f for f in found if f != "ether"]

    return found
