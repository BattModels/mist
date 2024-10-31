import re
from abc import abstractmethod, ABC
from typing import Iterable, Optional
import itertools
from rdkit import Chem

import smirk
import torch


def flatten(*iterables) -> list:
    return list(itertools.chain(*iterables))


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
    "chiral_tags": r"@{1,2}(?:[A-Z]{2}\d{1,2})?",
    "bracked_atom": r"\[[^]]+]",
    "charged_atom": r"\[[^\]]+?[+-]{1,2}\d{0,2}]",
    "chiral_center": r"\[[^\]]+?@[^\]]*?]",
    "aromatic_bracket_atom": r"\[[a-z]{1,2}[^\]]*?]",
}

ELEMENT_GROUPS = [
    ["Li", "Na", "K", "Rb", "Cs", "Fr"],  # Alkali Metals (Group 1)
    ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],  # Alkaline Earth Metals (Group 2)
    ["Sc", "Y", "Lu", "Lr"],  # Scandium Group (Group 3)
    ["Ti", "Zr", "Hf", "Rf"],  # Titanium Group (Group 4)
    ["V", "Nb", "Ta", "Db"],  # Vanadium Group (Group 5)
    ["Cr", "Mo", "W", "Sg"],  # Chromium Group (Group 6)
    ["Mn", "Tc", "Re", "Bh"],  # Manganese Group (Group 7)
    ["Fe", "Ru", "Os", "Hs"],  # Iron Group (Group 8)
    ["Co", "Rh", "Ir", "Mt"],  # Cobalt Group (Group 9)
    ["Ni", "Pd", "Pt", "Ds"],  # Nickel Group (Group 10)
    ["Cu", "Ag", "Au", "Rg"],  # Copper Group (Group 11)
    ["Zn", "Cd", "Hg", "Cn"],  # Zinc Group (Group 12)
    ["B", "Al", "Ga", "In", "Tl", "Nh"],  # Boron Group (Group 13)
    ["C", "Si", "Ge", "Sn", "Pb", "Fl"],  # Carbon Group (Group 14)
    ["N", "P", "As", "Sb", "Bi", "Mc"],  # Nitrogen Group (Group 15)
    ["O", "S", "Se", "Te", "Po", "Lv"],  # Chalcogens (Group 16)
    ["F", "Cl", "Br", "I", "At", "Ts"],  # Halogens (Group 17)
    ["Ne", "Ar", "Kr", "Xe", "Rn", "Rg"],  # Noble Gases (Group 18)
]

F_BLOCK = [
    [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ],
    [
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    ],
]

ELEMENT_FEATURES = {
    "alkali_metals": ELEMENT_GROUPS[0],
    "alkaline_earth_metals": ELEMENT_GROUPS[1],
    "nobel_gases": ["He"] + ELEMENT_GROUPS[16],
    "s_block": flatten(["H"], *ELEMENT_GROUPS[0:1]),
    "f_block": flatten(*F_BLOCK),
    "d_block": flatten(*ELEMENT_GROUPS[2:11]),
    "p_block": flatten(*ELEMENT_GROUPS[12:15]),
    "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te"],  # Commonly recognized per wiki
    "liquid_metals": ["Ga", "Hg", "Rb", "Cs", "Fr"],
    "semi_metalalic": ["As", "Sb", "Bi", "Sn"],
    "toxic_metals": ["As", "Be", "Cd", "Cr", "Pb", "Hg", "Ni"],  # Goyer & Clarkson 1996
    "radioactive": flatten(  # Elements that have no stable isotopes
        ["Tc", "Po", "At", "Rn", "Pm"],
        [group[-1] for group in ELEMENT_GROUPS],
        F_BLOCK[1],
    ),
    "opensmiles_aromatic": [
        "b",
        "c",
        "n",
        "o",
        "p",
        "s" "se",
        "as",
    ],
}

# Daylight Examples Marked (Daylight): https://daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
# rdkit.Chem.Lipinski from: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Lipinski.py
SMARTS_FEATURES = {
    "ketone": "[#6][CX3](=O)[#6]",  # Daylight
    "aldehyde": "[$([CX3H2](=O)),$([CX3H1](=O)[#6])]",  # Daylight, plus branch for Formaldehyde
    "carboxylic_acid": "[CX3](=O)[OX2H1]",  # Daylight
    "amid": "[$([NX3][CX3](=[OX1])[#6]),$(NC=O)]",  # Daylight, plus branch for Formamide
    "hydroxyl": "[OX2H]",  # Daylight
    "phenol": "[OH]c1ccccc1",
    "rotatable_bond": "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",  # rdkit.Chem.Lipinski
    "carboxyl_group": "[CX3]=[OX1]",  # Daylight
    "h_donor": "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]",  # rdkit.Chem.Lipinski
    "h_acceptor": "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]",  # rdkit.Chem.Lipinski
    "NH_or_OH_lipinski": "[#8H1,#7H1,#7H2,#7H3]",  # rdkit.Chem.Lipinski
    "NO_lipinski": "[#7H1,#7H2,#7H3]",  # rdkit.Chem.Lipinski
}


class Feature(ABC):
    requires_smirk = False

    def __init__(self, name: str, tokenizer: Optional[str] = None):
        self.name = name
        self.tokenzier = tokenizer or smirk.SmirkTokenizerFast()
        self.smirk_tokenizer = (
            smirk.SmirkTokenizerFast() if tokenizer is not None else self.tokenzier
        )

    def featurize(self, smi: str, encoding: Optional[dict] = None) -> torch.BoolTensor:
        encoding = encoding or self.tokenzier(smi, return_offsets_mapping=True)
        assert "offset_mapping" in encoding
        kwargs = self.preprocess(smi)
        return self._featurize(smi, encoding, **kwargs)

    @abstractmethod
    def _featurize(self, smi: str, encoding: dict, **kwargs) -> torch.BoolTensor:
        """Identify tokens in the input SMILES encoding expressing the feature"""

    def preprocess(self, smi: str) -> dict:
        """Shared preprocessing steps for all features to be provided to `self._featurize`
        Will be called once per feature class
        """
        return {}

    @classmethod
    @abstractmethod
    def from_named(cls, name: str, **kwargs) -> "Feature":
        """Create a feature from a named feature"""

    def align_tokens(self, encoding: dict, span: tuple[int, int]) -> Iterable[int]:
        """Identify tokens overlapping span"""
        token_offsets = encoding["offset_mapping"]
        start, end = span
        if end < start:
            return  # Non-matching span

        for i, t in enumerate(token_offsets):
            if t[1] <= t[0]:  # Token is empty
                continue

            # Spans don't include the end index
            elif not (end <= t[0] or t[1] <= start):
                print(f"token: {t}, span: {start}, {end}")
                yield i

    def align_embeddings(
        self, active: torch.BoolTensor, embedding: dict, other: dict
    ) -> torch.BoolTensor:
        if embedding == other:
            return active
        raise NotImplementedError()

    def onehot(self, indices: list[int], n: int) -> torch.BoolTensor:
        """Convert a list of indices to a one-hot encoding"""
        active = torch.zeros(n, dtype=torch.bool)
        if len(indices) > 0:
            print(indices)
            active[indices] = True
        return active


class RegexFeature(Feature):
    def __init__(self, name: str, regex: [str, re.Pattern], **kwargs):
        super().__init__(name, **kwargs)
        self.regex = re.compile(regex)

    @classmethod
    def from_named(cls, name: str, **kwargs):
        return cls(name, REGEX_FEATURES[name], **kwargs)

    def _featurize(self, smi: str, encoding: dict, **kwargs) -> torch.BoolTensor:
        n_groups = self.regex.groups
        active = []
        for m in self.regex.finditer(smi):
            if n_groups == 0:
                active.extend(self.align_tokens(encoding, m.span()))
            else:
                for i in range(1, n_groups + 1):
                    active.extend(self.align_tokens(encoding, m.span(i)))

        return self.onehot(active, len(encoding["input_ids"]))


class ElementFeature(Feature):
    def __init__(self, name: str, elements: list[str], **kwargs):
        super().__init__(name, **kwargs)
        self.elements = list(set(elements))
        self.element_ids = torch.tensor(
            [
                self.smirk_tokenizer.encode(f"[{e}]", add_special_tokens=False)[1]
                for e in self.elements
            ]
        ).reshape(-1, 1)

    @classmethod
    def from_named(cls, name: str, **kwargs) -> "ElementFeature":
        return cls(name, ELEMENT_FEATURES[name], **kwargs)

    def preprocess(self, smi: str) -> dict:
        return {
            "smirk_encoding": self.smirk_tokenizer(smi, return_offsets_mapping=True)
        }

    def _featurize(
        self, smi: str, encoding: dict, smirk_encoding: dict
    ) -> torch.BoolTensor:
        enc = torch.tensor(smirk_encoding["input_ids"])
        active = enc.eq(self.element_ids).any(dim=0)
        return self.align_embeddings(active, encoding, smirk_encoding)


class RdkitFeature(Feature):
    atomwise = re.compile(r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p")

    def preprocess(self, smi: str) -> dict:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        s_flags = Chem.SanitizeFlags.SANITIZE_NONE
        s_flags |= Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        s_flags |= Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        s_flags |= Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        s_flags |= Chem.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, s_flags)
        atom_spans = [m.span() for m in self.atomwise.finditer(smi)]

        # Validate rdkit -> smi mapping
        for idx, atom in enumerate(mol.GetAtoms()):
            smi_atom = smi[atom_spans[idx][0] : atom_spans[idx][1]]
            smi_atom_mol = Chem.MolFromSmiles(smi_atom, sanitize=False)
            Chem.SanitizeMol(smi_atom_mol, s_flags)

            assert atom.GetSymbol() == smi_atom_mol.GetAtomWithIdx(0).GetSymbol()

            # atom_smi = atom.GetSmarts()
            # smi_atom_rdkit = Chem.MolToSmiles(smi_atom_mol)
            # assert (
            #     atom_smi == Chem.MolToSmiles(smi_atom_mol)
            # ), f"Expected {atom_smi} and {smi_atom_rdkit} to match. Input atom: {smi_atom}"

        return {"rdkit_molecule": mol, "atom_spans": atom_spans}

    def align_atoms(
        self,
        atom_idx: int,
        encoding: dict,
        atom_spans: list[tuple[int, int]],
    ) -> Iterable[int]:
        """Map atom indices to token indices"""
        span = atom_spans[atom_idx]
        return self.align_tokens(encoding, span)


class SMARTSFeature(RdkitFeature):
    def __init__(self, name: str, smarts: str, **kwargs):
        super().__init__(name, **kwargs)
        self.smarts = Chem.MolFromSmarts(smarts)

    @classmethod
    def from_named(cls, name: str, **kwargs):
        return cls(name, SMARTS_FEATURES[name], **kwargs)

    def _featurize(
        self,
        smi: str,
        encoding: dict,
        atom_spans: list[tuple[int, int]],
        rdkit_molecule: Chem.Mol,
    ) -> torch.BoolTensor:
        # TODO: Handle Bonds
        matches = rdkit_molecule.GetSubstructMatches(self.smarts)
        atom_indices = set(flatten(*matches))
        active: list[int] = []
        for idx in atom_indices:
            active.extend(self.align_atoms(idx, encoding, atom_spans))
        return self.onehot(active, len(encoding["input_ids"]))
