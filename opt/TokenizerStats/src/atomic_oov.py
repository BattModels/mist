import json
import logging
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice, chain
from typing import Optional, Any

import selfies
from datasets import load_dataset
from mendeleev import element

from electrolyte_fm.utils.tokenizer import PreTrainedTokenizerBase, load_tokenizer
from opt.build_vocab import (
    ALIPHATIC_ORGANIC,
    AROMATIC_ORGANIC,
    AROMATIC_SYMBOLS,
    CHIRAL,
    ELEMENT_SYMBOLS,
    BONDS,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def batched(iterable, n):
    # backport itertools.batched from 3.12
    # Source: https://docs.python.org/3.12/library/itertools.html#itertools.batched
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


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


def include_isotopes(base=None):
    """Add isotopes variants to an existing iterator"""
    for atom in base or elements():
        if atom.symbol == "*":
            yield atom
            continue

        atom.isotope = None
        yield deepcopy(atom)
        for isotope in element(atom.symbol).isotopes:
            atom.isotope = isotope.mass_number
            yield deepcopy(atom)


def include_chirality(base=None, chirality=CHIRAL):
    """Add chiral variants to a base atomic iterator"""
    for atom in base or elements():
        atom.chiral = None
        yield deepcopy(atom)
        for chiral in chirality:
            atom.chiral = chiral
            yield deepcopy(atom)


def include_charge(base=None):
    """Add charge variants to the base atomic iterator based on possible oxidation states"""
    for atom in base or elements():
        atom.charge = None
        yield deepcopy(atom)
        oxidation_states = element(atom.symbol).oxidation_states()
        for oxidation in oxidation_states:
            atom.charge = oxidation
            yield deepcopy(atom)


def bonds(element="C"):
    """Iterator over all bonds"""
    for bond in BONDS:
        yield element + bond + element


def rings():
    """Iterator over all permissible carbon rings"""
    for ring in range(0, 100):
        if ring < 10:
            yield f"C{ring:d}CCCCC{ring:d}"
        yield f"C%{ring:02d}CCCCC%{ring:02d}"


def fullerene():
    """Iterator over Carbon Fullerenes from C20 to C720 from
    https://nanotube.msu.edu/fullerene/fullerene-isomers.html
    """
    with open(Path(__file__).parent.parent.joinpath("fullerene.smi")) as fid:
        for line in fid.readlines():
            yield line.strip()


def tokenize(tok: PreTrainedTokenizerBase, batch):
    out = tok(batch["text"])
    out["decode"] = tok.batch_decode(out["input_ids"], skip_special_tokens=True)
    out["text"] = batch
    return {"oov": [o != i for o, i in zip(out["decode"], batch["text"])]}


MOLNET_DATASET = {
    "QM8": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
    "QM9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "ESOL": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "FreeSolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "Lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "MUV": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "HIV": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "BACE": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "BBBP": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "Tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "ToxCast": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
    "SIDER": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "ClinTox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
}


def molecularnet(subset):
    url = MOLNET_DATASET[subset]
    ds = load_dataset(
        "csv",
        data_files=[url],
        split="train",
        keep_in_memory=False,
    )
    for batch in ds.iter(batch_size=100):
        try:
            yield from batch["smiles"]
        except KeyError:
            yield from batch["mol"]


ATOM_DATASETS = {
    "elements": elements,
    "rings": rings,
    "bonds": bonds,
    "fullerenes": fullerene,
    "isotopes": lambda: include_isotopes(elements()),
    "chiral_elements": lambda: include_chirality(elements()),
    "chiral_isotopes": lambda: include_chirality(include_isotopes()),
    "charged_elements": lambda: include_charge(elements()),
    "charged_isotops": lambda: include_isotopes(include_charge(elements())),
    "charged_chiral_isotopes": lambda: include_chirality(
        include_isotopes(include_charge(elements()))
    ),
}

for subset in MOLNET_DATASET.keys():
    name = f"MoleculeNet/{subset}"
    ATOM_DATASETS[name] = lambda subset=subset: molecularnet(subset)

TOKENIZERS = {
    "smiles": [
        "character",
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
        "./smirk-gpe-50k-mb-ss",
        "./smirk-gpe-50k-nmb-ss",
        "./smirk-gpe-small-50k-mb-ss",
        "google/gemma-7b",
        "Xenova/gpt-4o",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3-8B",
    ],
    "selfies": [
        "HUBioDataLab/SELFormer",
    ],
}


def build_atom_generator(name):
    base = elements()
    for atom in include_isotopes(base):
        yield {"text": str(atom)}


def safe_selfies(iter):
    """Skip molecules that can not be encoded as SELFIES"""
    for smi in iter:
        try:
            yield selfies.encoder(str(smi), strict=False)
        except selfies.exceptions.EncoderError:
            LOG.warn("failed to encode %s as a SELFIES", smi)


def tabulate_tokenizer(
    tok: PreTrainedTokenizerBase,
    datasets: dict[str, Any],
    encoding: str = "smiles",
) -> dict:
    out = dict()
    for ds_name, iter in datasets.items():
        unk_token_id = tok.unk_token_id
        nobs = 0
        n_oov = 0
        oov_samples = set()

        # Encode molecules
        if encoding == "smiles":
            ds = (str(x) for x in iter())
        elif encoding == "selfies":
            ds = safe_selfies(iter())
        else:
            raise ValueError(f"Unknown encoding {encoding}")

        for batch in batched(ds, 1000):
            batch_input_ids = tok(batch)["input_ids"]
            # Tests are in test/test_tokenizer.py::test_oov_tokens
            # to ensure that unk_token_id is correctly emitted by
            # tokenizers
            for smi, obs in zip(batch, batch_input_ids):
                if unk_token_id in obs:
                    n_oov += 1
                    if len(oov_samples) < 20:
                        oov_samples.add(smi)
                nobs += 1
            LOG.info("%s - %s: finished %d", name, ds_name, nobs)
            break

        out[ds_name] = {
            "nobs": nobs,
            "oov": n_oov,
            "oov_samples": list(oov_samples),
        }
        LOG.info("%s - %s: %d/%d", name, ds_name, n_oov, nobs)
    return out


if __name__ == "__main__":
    out = defaultdict(lambda: defaultdict(dict))
    for encoding, tokenizers in TOKENIZERS.items():
        if encoding != "selfies":
            pass
            # continue
        for name in tokenizers:
            LOG.info("processing %s", name)
            tok = load_tokenizer(name)
            out[name] = tabulate_tokenizer(tok, ATOM_DATASETS, encoding)

    with open("stats-atomic.json", "w") as fid:
        json.dump(out, fid)
