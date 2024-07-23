"""
Not really a test, as vocab files are checked into source control.
More of a double check that the right files were checked in
"""

import json
from importlib.resources import files

from opt.build_vocab import build_selfies_alphabet, build_smiles_alphabet, build_vocab


def check_vocab(file: str, alphabet: set):
    with open(file, "r") as fid:
        src_vocab = json.load(fid)

    # Reconstruct generated vocab
    gen_vocab = build_vocab(alphabet)
    assert src_vocab == gen_vocab


def test_smiles_vocab():
    check_vocab(files("smirk").joinpath("vocab_smiles.json"), build_smiles_alphabet())


def test_selfies_vocab():
    check_vocab(files("smirk").joinpath("vocab_selfies.json"), build_selfies_alphabet())
