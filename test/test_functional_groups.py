import pytest
from rdkit import RDLogger
from electrolyte_fm.interpretibility.functional_groups import identify_functional_groups

# Silence RDKit warnings for clean test output
RDLogger.DisableLog("rdApp.*")


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Hydrocarbons
        ("CC", ["alkane"]),  # ethane
        ("C=C", ["alkene"]),  # ethene
        ("C#C", ["alkyne"]),  # ethyne
        ("c1ccccc1", ["arene"]),  # benzene
        # Simple oxygen heteroatomics
        ("CCO", ["alcohol"]),  # ethanol
        ("COC", ["ether"]),  # methoxymethane
        ("C1CO1", ["epoxide"]),  # ethene oxide
        # Halogen heteroatomics
        ("CCCl", ["haloalkane"]),  # chloroethane
        # Carbonyl compounds
        ("CC=O", ["aldehyde"]),  # acetaldehyde
        ("CCC(=O)C", ["ketone"]),  # 2-butanone
        ("CC(=O)O", ["carboxylic acid"]),  # acetic acid
        ("CC(=O)OC(=O)C", ["acid anhydride"]),  # acetic anhydride
        ("CC(=O)OC", ["ester"]),  # methyl acetate
        ("CC(=O)N", ["amide"]),  # acetamide
        ("CC(=O)Cl", ["acyl halide"]),  # acetyl chloride
        # Nitrogen-based
        ("CN", ["amine"]),  # methylamine
        ("CC#N", ["nitrile"]),  # acetonitrile
        ("C=N", ["imine"]),  # methanimine
        ("CN=C=O", ["isocyanate"]),  # ethyl isocyanate
        ("CCN=NC", ["azo compound"]),  # dimethylazo
        # Sulfur-based
        ("CS", ["thiol"]),  # methanethiol
        # Electrolytes
        ("COCCOC", ["ether"]),
        ("COC(=O)OC", ["carbonate ester"]),
        # Mixed functional groups
        ("CC(=O)OC", ["ester"]),  # methyl acetate
        ("CN(C)C", ["amine"]),  # trimethylamine (tertiary amine)
        ("CC(C)Cl", ["haloalkane"]),  # 2-chloropropane
        ("CC(=O)OC(=O)C", ["acid anhydride"]),  # hybrid match
        ("COc1ccccc1", ["ether", "arene"]),  # anisole
        ("CCC(C)S", ["thiol"]),
        ("CC(=O)OCc1ccccc1", ["ester", "arene"]),  # benzyl acetate
    ],
)
def test_functional_group_detection(smiles, expected):
    found = identify_functional_groups(smiles)
    assert set(found) == set(expected), f"Expected {expected}, got {found}"
