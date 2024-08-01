import json
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product, chain, batched
from dataclasses import dataclass
from typing import Optional
from electrolyte_fm.utils.tokenizer import load_tokenizer, PreTrainedTokenizerBase
from opt.build_vocab import (
    ALIPHATIC_ORGANIC,
    AROMATIC_ORGANIC,
    AROMATIC_SYMBOLS,
    ELEMENT_SYMBOLS,
    CHIRAL,
)


from mendeleev import Element, element, get_all_elements
from datasets import IterableDataset

logging.basicConfig(level=logging.INFO)


@dataclass
class Atom:
    symbol: str
    isotope: Optional[str] = None
    chiral: Optional[str] = None
    charge: Optional[int] = None
    hcount: Optional[int] = None
    aromatic: bool = False

    @property
    def isplain(self):
        return (
            self.isotope is None
            and self.chiral is None
            and self.charge is None
            and self.hcount is None
        )

    @property
    def is_bracket_atom(self):
        if not self.isplain:
            return True
        if self.symbol in ALIPHATIC_ORGANIC:
            return False
        if self.symbol in AROMATIC_ORGANIC:
            return False
        return True

    def __str__(self):
        s = self.symbol if not self.aromatic else self.symbol.lower()
        if self.is_bracket_atom:
            if isotope := self.isotope:
                s = f"{isotope:d}{s}"
            if chiral := self.chiral:
                s = f"{s}{chiral}"
            if hcount := self.hcount:
                if hcount == 1:
                    s = f"{s}H"
                elif hcount > 1:
                    s = f"{s}H{hcount:d}"
            if charge := self.charge:
                if charge != 0:
                    s = f"{s}{charge:+d}"

            s = f"[{s}]"
        return s


def elements(include_wildcard=False):
    """Iterator over all plain (non-isotope, neural, non-chiral, zero explicit hydrogen) elements.
    By default does not include the wildcard `*` atom
    """
    non_bracket_atoms = set(chain(ALIPHATIC_ORGANIC, AROMATIC_ORGANIC))
    if include_wildcard:
        non_bracket_atoms.add("*")
    yield from (Atom(sym) for sym in ALIPHATIC_ORGANIC)
    yield from (Atom(sym.upper(), aromatic=True) for sym in AROMATIC_ORGANIC)
    for symbol in ELEMENT_SYMBOLS:
        if symbol in non_bracket_atoms:
            continue
        yield Atom(symbol)

    assert (set(AROMATIC_SYMBOLS) - set(AROMATIC_ORGANIC)) == set(["se", "as"])
    yield Atom("Se", aromatic=True)
    yield Atom("As", aromatic=True)


def include_isotopes(base=elements()):
    """Add isotopes variants to an existing iterator"""
    for atom in base:
        if atom.symbol == "*":
            yield atom
            continue

        for isotope in element(atom.symbol).isotopes:
            a = deepcopy(atom)
            a.isotope = isotope.mass_number
            yield a


def include_chirality(base=elements(), chirality=CHIRAL):
    """Add chirality vaiants to a base atomic iterator"""
    for atom in base:
        a = deepcopy(atom)
        a.chiral = None
        yield a
        for chiral in chirality:
            a = deepcopy(a)
            a.chiral = chiral
            yield a


def include_charge(base=elements()):
    """Add charge variants to the base atomic iterator based on possible oxidation states"""
    for atom in base:
        a = deepcopy(atom)
        a.charge = None
        yield a
        for oxidation in element(a.symbol).oxidation_states():
            a = deepcopy(a)
            a.charge = oxidation
            yield a


def tokenize(tok: PreTrainedTokenizerBase, batch):
    out = tok(batch["text"])
    out["decode"] = tok.batch_decode(out["input_ids"], skip_special_tokens=True)
    out["text"] = batch
    return {"oov": [o != i for o, i in zip(out["decode"], batch["text"])]}


ATOM_DATASETS = {
    "elements": elements,
    "isotopes": lambda: include_isotopes(elements()),
    "chiral_elements": lambda: include_chirality(elements()),
    "chiral_isotopes": lambda: include_chirality(include_isotopes()),
    "charged_elements": lambda: include_charge(elements()),
    "charged_isotops": lambda: include_charge(include_isotopes(elements())),
    "charged_chiral_isotopes": lambda: include_chirality(
        include_charge(include_isotopes(elements()))
    ),
}

TOKENIZERS = [
    "smirk",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "SmilesPE/SPE_ChEMBL",
    "devalab/molgpt-moses",
    "devalab/molgpt-guacamol",
    "MolecularAI/Chemformer",
    "MolecularAI/Chemformer-downstream",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
    "sagawa/ReactionT5-yield-prediction",
    "rxn4chemistry/rxn_yields",
    "rxn4chemistry/rxnfp",
    "ChangwenXu98/TransPolymer",
]


def build_atom_generator(name):
    base = elements()
    for atom in include_isotopes(base):
        yield {"text": str(atom)}


if __name__ == "__main__":
    out = defaultdict(lambda: defaultdict(dict))
    for name in TOKENIZERS:
        for ds_name, iter in ATOM_DATASETS.items():
            logging.info("processing %s - %s", name, ds_name)
            tok = load_tokenizer(name)
            unk_token_id = tok.unk_token_id
            nobs = 0
            n_oov = 0

            for batch in batched((str(x) for x in iter()), 1000):
                input_ids = tok(batch)["input_ids"]
                # Tests are in test/test_tokenizer.py::test_oov_tokens
                # to ensure that unk_token_id is correctly emitted by
                # tokenizers
                for unk_token_id in input_ids:
                    n_oov += 1
                nobs += 1

            out[name][ds_name] = {"nobs": nobs, "oov": n_oov}

    with open("stats-atomic.json", "w") as fid:
        json.dump(out, fid)

    print(json.dumps(out))
