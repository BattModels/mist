# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "rdkit"
# ]
# ///

import json
import pandas as pd
import re
from rdkit import Chem


def isotope_to_smiles(isotope: str) -> str:
    """
    Convert isotope label like 'actinium-227' or 'carbon-14' into isotope SMILES.
    SMILES syntax: [<mass><element>] e.g. [227Ac], [14C].
    """
    match = re.match(r"([a-zA-Z]+)-(\d+)(m\d+)?", isotope)
    if not match:
        raise ValueError(f"Could not parse symbol for: {isotope}")
    periodic_symbols = {
        "actinium": "Ac",
        "aluminium": "Al",
        "americium": "Am",
        "antimony": "Sb",
        "argon": "Ar",
        "arsenic": "As",
        "astatine": "At",
        "barium": "Ba",
        "berkelium": "Bk",
        "beryllium": "Be",
        "bismuth": "Bi",
        "bohrium": "Bh",
        "boron": "B",
        "cadmium": "Cd",
        "caesium": "Cs",
        "calcium": "Ca",
        "californium": "Cf",
        "carbon": "C",
        "cerium": "Ce",
        "chlorine": "Cl",
        "chromium": "Cr",
        "cobalt": "Co",
        "copernicium": "Cn",
        "copper": "Cu",
        "curium": "Cm",
        "darmstadtium": "Ds",
        "dubnium": "Db",
        "dysprosium": "Dy",
        "einsteinium": "Es",
        "erbium": "Er",
        "europium": "Eu",
        "fermium": "Fm",
        "fluorine": "F",
        "francium": "Fr",
        "flerovium": "Fl",
        "gadolinium": "Gd",
        "gallium": "Ga",
        "germanium": "Ge",
        "gold": "Au",
        "hafnium": "Hf",
        "hassium": "Hs",
        "helium": "He",
        "holmium": "Ho",
        "hydrogen": "H",
        "indium": "In",
        "iodine": "I",
        "iridium": "Ir",
        "iron": "Fe",
        "krypton": "Kr",
        "livermorium": "Lv",
        "lanthanum": "La",
        "lawrencium": "Lr",
        "lead": "Pb",
        "lithium": "Li",
        "lutetium": "Lu",
        "magnesium": "Mg",
        "manganese": "Mn",
        "meitnerium": "Mt",
        "mendelevium": "Md",
        "mercury": "Hg",
        "molybdenum": "Mo",
        "moscovium": "Mc",
        "neodymium": "Nd",
        "nihonium": "Nh",
        "neon": "Ne",
        "neptunium": "Np",
        "nickel": "Ni",
        "niobium": "Nb",
        "nitrogen": "N",
        "nobelium": "No",
        "oganesson": "Og",
        "osmium": "Os",
        "oxygen": "O",
        "palladium": "Pd",
        "phosphorus": "P",
        "platinum": "Pt",
        "plutonium": "Pu",
        "polonium": "Po",
        "potassium": "K",
        "praseodymium": "Pr",
        "promethium": "Pm",
        "protactinium": "Pa",
        "radium": "Ra",
        "radon": "Rn",
        "rhenium": "Re",
        "rhodium": "Rh",
        "roentgenium": "Rg",
        "rubidium": "Rb",
        "ruthenium": "Ru",
        "rutherfordium": "Rf",
        "samarium": "Sm",
        "scandium": "Sc",
        "seaborgium": "Sg",
        "selenium": "Se",
        "silicon": "Si",
        "silver": "Ag",
        "sodium": "Na",
        "strontium": "Sr",
        "sulfur": "S",
        "tantalum": "Ta",
        "tennessine": "Ts",
        "technetium": "Tc",
        "tellurium": "Te",
        "terbium": "Tb",
        "thallium": "Tl",
        "thorium": "Th",
        "thulium": "Tm",
        "tin": "Sn",
        "titanium": "Ti",
        "tungsten": "W",
        "uranium": "U",
        "vanadium": "V",
        "xenon": "Xe",
        "ytterbium": "Yb",
        "yttrium": "Y",
        "zinc": "Zn",
        "zirconium": "Zr",
    }

    element, mass, meta = match.groups()
    symbol = periodic_symbols.get(element.lower())
    smiles = f"[{mass}{symbol}]"
    return smiles


def get_protons_neutrons(row: str):
    """
    Extract the number of protons (atomic number) and neutrons
    from isotope SMILES like '[227Ac]' or '[14C]'.
    """
    smiles = row["smiles"]
    # Handle metastable states e.g. [227Acm1]
    match = re.match(r"\[(\d+)([A-Z][a-z]?)(m\d+)?\]", smiles)
    if not match:
        raise ValueError(f"Invalid isotope SMILES: {smiles}")

    mass_number = int(match.group(1))
    element = match.group(2)

    atom = Chem.Atom(element)
    protons = atom.GetAtomicNum()
    neutrons = mass_number - protons

    return pd.Series({"neutrons": neutrons, "protons": protons})


if __name__ == "__main__":
    results = []
    with open("Radioactive-Isotope-Half-Lives.json", "r") as f:
        records = json.load(f)
    for r in records[1][1:]:
        name = r[1][-1][1:-1]
        half_life = r[2][-1]
        log_time = r[3][-1]
        if not (name.endswith("m") or name.endswith("m1") or name.endswith("m2")):
            results.append({"name": name, "half_life": half_life, "log_time": log_time})
    df = pd.DataFrame.from_records(results)
    df["smiles"] = df.name.apply(isotope_to_smiles)
    df[["neutrons", "protons"]] = df.apply(get_protons_neutrons, axis=1)
    df.to_csv("isotope_half_lives.csv")
