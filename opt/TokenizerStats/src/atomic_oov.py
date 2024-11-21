import argparse
import concurrent.futures
import itertools
import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import selfies
from build_vocab import (
    ALIPHATIC_ORGANIC,
    AROMATIC_ORGANIC,
    AROMATIC_SYMBOLS,
    BONDS,
    CHIRAL,
    ELEMENT_SYMBOLS,
)
from datasets import load_dataset
from mendeleev import element

from electrolyte_fm.utils.tokenizer import PreTrainedTokenizerBase, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d]: %(message)s",
)
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
    chirality = list(chirality)
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


def chiral_extended():
    return itertools.chain(
        deepcopy(CHIRAL),
        ["@TH1", "@TH2"],
        ["@AL1", "@AL2"],
        ["@SP1", "@SP2", "@SP3"],
        (f"@TB{d}" for d in range(1, 21)),
        (f"@OH{d}" for d in range(1, 31)),
    )


DATASETS: dict[str, Callable[[], Iterable[Atom | str]]] = {
    "elements": elements,
    "rings": rings,
    "bonds": bonds,
    "fullerenes": fullerene,
    "isotopes": lambda: include_isotopes(elements()),
    "chiral_elements": lambda: include_chirality(elements()),
    "extended_chiral_elements": lambda: include_chirality(
        elements(),
        chirality=chiral_extended(),
    ),
    "chiral_isotopes": lambda: include_chirality(include_isotopes()),
    "charged_elements": lambda: include_charge(elements()),
    "charged_isotops": lambda: include_isotopes(include_charge(elements())),
    "charged_chiral_isotopes": lambda: include_chirality(
        include_isotopes(include_charge(elements()))
    ),
    "extended_charged_chiral_isotopes": lambda: include_chirality(
        include_isotopes(include_charge(elements())),
        chirality=chiral_extended(),
    ),
    "tmQM": lambda: tmqm_dataset(),
}

for subset in MOLNET_DATASET.keys():
    name = f"MoleculeNet/{subset}"
    DATASETS[name] = lambda subset=subset: molecularnet(subset)


def tmqm_dataset(path: Optional[str] = None):
    if path is not None:
        path = Path(path)
    else:
        path = Path(__file__).parent.parent.parent.joinpath("tmQM")

    ds = load_dataset(
        "arrow",
        name="tmQM",
        data_files=[str(path.joinpath("data/*/*.arrow"))],
    )
    for obs in ds["train"]:
        yield obs["smiles"]


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
            LOG.debug("failed to encode %s as a SELFIES", smi)
            yield None


def filter_and_count_nones(iterator):
    nones = 0
    filtered_iterator = []
    for item in iterator:
        if item is None:
            nones += 1
        else:
            filtered_iterator.append(item)
    return filtered_iterator, nones


def tabulate_tokenizer(
    tok: PreTrainedTokenizerBase,
    dataset: Callable[[], Iterable[Atom | str]],
    encoding: str = "smiles",
) -> dict:
    unk_token_id = tok.unk_token_id
    nobs = 0
    n_oov = 0
    n_failed_encode = 0
    oov_samples = set()

    # Encode molecules
    if encoding == "smiles":
        ds = (str(x) for x in dataset())
    elif encoding == "selfies":
        ds = safe_selfies(dataset())

    for batch in batched(ds, 1000):
        batch, failed_encode = filter_and_count_nones(batch)
        n_failed_encode += failed_encode

        # Fast tokenizers can be used directly
        if hasattr(tok, "is_fast") and tok.is_fast:
            batch_input_ids = tok(batch)["input_ids"]
        else:
            batch_input_ids = [tok(smi)["input_ids"] for smi in batch]

        # Tests are in test/test_tokenizer.py::test_oov_tokens
        # to ensure that unk_token_id is correctly emitted by
        # tokenizers
        for smi, obs in zip(batch, batch_input_ids):
            if unk_token_id in obs:
                n_oov += 1
                if len(oov_samples) < 20:
                    oov_samples.add(smi)
            nobs += 1

    return {
        "nobs": nobs,
        "oov": n_oov,
        "oov_samples": list(oov_samples),
        "failed_encode": n_failed_encode,
    }


def process_tokenizer(dataset_name: str, tokenizer: dict[str, str]) -> dict:
    tok = load_tokenizer(tokenizer["name_or_path"])
    dataset = DATASETS[dataset_name]
    LOG.info("processing %s for %s", dataset_name, tokenizer["name"])
    return tabulate_tokenizer(tok, dataset, tokenizer["encoding"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name_or_path", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("-d", "--dataset", type=str, default=None, action="append")
    parser.add_argument("--output", type=argparse.FileType("w"), default="-")
    args = parser.parse_args()

    # Lookup tokenizer information
    tokenizer = {
        "name": args.name,
        "name_or_path": args.name_or_path,
        "encoding": args.encoding,
    }
    tokenizers = Path(__file__).parent.parent.joinpath("tokenizers.json").read_text()
    for tokenizer in json.loads(tokenizers):
        if tokenizer["name_or_path"] == args.name_or_path:
            tokenizer["name"] = args.name or tokenizer["name"]
            tokenizer["encoding"] = args.encoding or tokenizer["encoding"]
            break

    # Process datasets in parallel
    datasets = args.dataset or DATASETS.keys()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_tokenizer, dataset, tokenizer): dataset
            for dataset in datasets
        }

        out = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                out[name] = future.result()
            except Exception as e:
                LOG.error("Error processing %s: %s", name, e)

    # Dump results
    json.dump(out, args.output)
