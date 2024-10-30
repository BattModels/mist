import re
from abc import abstractmethod
from typing import Iterable, Optional
import itertools

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
SMARTS_FEATURES = {
    "ketone": "[#6][CX3](=O)[#6]",  # Daylight
    "aldehyde": "[CX3H1](=O)[#6]",  # Daylight
    "carboxylic_acid": "[CX3](=O)[OX2H1]",  # Daylight
    "amid": "[NX3][CX3](=[OX1])[#6]",  # Daylight
    "hydroxyl": "[OX2H]",  # Daylight
    "phenol": "[OX2H][cX3]:[c]",  # Daylight
    "rotatable_bond": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",  # Daylight
}


class Feature:
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

        kwargs = {}
        if self.requires_smirk:
            smirk_encoding = self.smirk_tokenizer(
                smi,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            kwargs["smirk_encoding"] = smirk_encoding

        return self._featurize(smi, encoding, **kwargs)

    @abstractmethod
    def _featurize(self, smi: str, encoding: dict, **kwargs) -> torch.BoolTensor:
        """Identify tokens in the input SMILES encoding expressing the feature"""

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
    requires_smirk = True

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

    def _featurize(
        self, smi: str, encoding: dict, smirk_encoding: dict
    ) -> torch.BoolTensor:
        enc = torch.tensor(smirk_encoding["input_ids"])
        active = enc.eq(self.element_ids).any(dim=0)
        return self.align_embeddings(active, encoding, smirk_encoding)
