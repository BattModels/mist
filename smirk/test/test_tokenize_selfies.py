import json
from copy import deepcopy
from importlib.resources import files

import pytest

from smirk.smirk import SmirkTokenizer


@pytest.fixture
def tokenizer():
    VOCAB_FILE = files("smirk").joinpath("vocab_selfies.json")
    assert VOCAB_FILE.is_file()
    return SmirkTokenizer.from_vocab(str(VOCAB_FILE), is_smiles=False)


@pytest.fixture
def selfie_strings():
    return [
        "[C][N][=C][=O]",
        "[C][N][C][=N][C][=C][Ring1][Branch1][C][=Branch1][C][=O][N][Branch1][=Branch2][C][=Branch1][C][=O][N][Ring1][Branch2][C][C]",
    ]


def check_encode(tok, x):
    img = tok.decode(tok.encode(x)["input_ids"])
    assert img == x


def test_pretokenize(tokenizer):
    splits = tokenizer.pretokenize("[C][N][=C][=O]")
    assert splits == [
        "[",
        "C",
        "]",
        "[",
        "N",
        "]",
        "[",
        "=",
        "C",
        "]",
        "[",
        "=",
        "O",
        "]",
    ]
    assert len(splits) == 14


def test_image(tokenizer, selfie_strings):
    check_encode(tokenizer, "[C][N][=C][=O]")
    assert len(tokenizer.encode("[C][N][=C][=O]")["input_ids"]) == 14
    for x in selfie_strings:
        check_encode(tokenizer, x)


def test_encode(tokenizer):
    selfie = "[C][N][C][=N][C][=C][Ring1][Branch1][C][=Branch1][C][=O][N][Branch1][=Branch2][C][=Branch1][C][=O][N][Ring1][Branch2][C][C]"
    emb = tokenizer.encode(selfie)
    out = tokenizer.decode(emb["input_ids"])
    assert out == selfie
