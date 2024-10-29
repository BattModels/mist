import re
import torch
import smirk
from rdkit import Chem

# fmt: off
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
# fmt: on

BOND_TYPES = ["-", "=", "#", ":", "$", "/", "\\"]

REGEX_FEATURES = {
    "chiral_tags": re.compile(r"@{1,2}(?:[A-Z]{2}\d{1,2})?"),
    "bracked_atom": re.compile(r"\[[^]]+]"),
    "charged_atom": re.compile(r"\[[^\]]+?[+-]{1,2}\d{0,2}]"),
    "chiral_center": re.compile(r"\[[^\]]+?@[^\]]*?]"),
    "aromatic_atom": re.compile(r"(?:b|c|o|p|se?|as)|(?:\[[a-z]{1,2}[^\]]+?])"),
}

ELEMENT_GROUPS = [
    ["Li", "Na", "K", "Rb", "Cs", "Fr"],  # Alkali Metals (Group 1)
    ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],  # Alkaline Earth Metals (Group 2)
    ["Sc", "Y", "Lu", "Lr"],  # Scandium Group (Group 3)
    ["Ti", "Zr", "Hf", "Rf"],  # Titanium Group (Group 4)
    ["V", "Nb", "Ta", "Db"],   # Vanadium Group (Group 5)
    ["Cr", "Mo", "W", "Sg"],   # Chromium Group (Group 6)
    ["Mn", "Tc", "Re", "Bh"],  # Manganese Group (Group 7)
    ["Fe", "Ru", "Os", "Hs"],  # Iron Group (Group 8)
    ["Co", "Rh", "Ir", "Mt"],  # Cobalt Group (Group 9)
    ["Ni", "Pd", "Pt", "Ds"],  # Nickel Group (Group 10)
    ["Cu", "Ag", "Au", "Rg"],  # Copper Group (Group 11)
    ["Zn", "Cd", "Hg", "Cn"],  # Zinc Group (Group 12)
    ["B", "Al", "Ga", "In", "Tl", "Nh"],   # Boron Group (Group 13)
    ["C", "Si", "Ge", "Sn", "Pb", "Fl"],   # Carbon Group (Group 14)
    ["N", "P", "As", "Sb", "Bi", "Mc"],    # Nitrogen Group (Group 15)
    ["O", "S", "Se", "Te", "Po", "Lv"],    # Chalcogens (Group 16)
    ["F", "Cl", "Br", "I", "At", "Ts"]     # Halogens (Group 17)
    # Lanthanides (f-block)
    ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
    # Actinides (f-block)
    ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
]

ELEMENT_SETS = {
    "alkali_metals": ELEMENT_GROUPS[0],
    "alkaline_earth_metals": ELEMENT_GROUPS[1],
    "nobel_gases": ["He"] + ELEMENT_GROUPS[17],
    "s_block": ["H"] + ELEMENT_GROUPS[1] + ELEMENT_GROUPS[1],
    "f_block": [*ELEMENT_GROUPS[17:18]],
    "d_block": [*ELEMENT_GROUPS[2:11]],
    "p_block": [*ELEMENT_GROUPS[12:16]],
    "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te"], # Commonly recognized per wiki
    "liquid_metals": ["Ga", "Hg", "Rb", "Cs", "Fr"],
    "semi_metalalic": ["As", "Sb", "Bi", "Sn"],
    "toxic_metals": ["As", "Be", "Cd", "Cr", "Pb", "Hg", "Ni"], # Goyer & Clarkson 1996
}

# Daylight Examples Marked (Daylight): https://daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
SMARTS_FEATURES = {
    "ketone": "[#6][CX3](=O)[#6]", # Daylight
    "aldehyde": "[CX3H1](=O)[#6]", # Daylight
    "carboxylic_acid": "[CX3](=O)[OX2H1]", # Daylight
    "amid": "[NX3][CX3](=[OX1])[#6]", # Daylight
    "hydroxyl": "[OX2H]", # Daylight
    "phenol": "[OX2H][cX3]:[c]", # Daylight
    "rotable_bond": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]", # Daylight
}
