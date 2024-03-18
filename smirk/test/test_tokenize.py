import smirk
import pytest
import json
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
from transformers.data import DataCollatorForLanguageModeling


@pytest.fixture
def tokenizer():
    VOCAB_FILE = Path(__file__).parent.parent.joinpath("vocab.json").resolve()
    assert VOCAB_FILE.is_file()
    return smirk.SmirkTokenizer(str(VOCAB_FILE), padding=False)


@pytest.fixture
def smile_strings():
    return [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
    ]


def test_encode(tokenizer, smile_strings):
    smile = "COCCC(=O)N1CCN(C)C(C2=CN([C@@H](C)C3=CC=C(C(F)(F)F)C=C3)N=N2)C1"
    emb = tokenizer.encode(smile)
    assert len(emb.ids) == 62
    smile_out = tokenizer.decode(emb.ids)
    assert smile_out == smile


def test_encode_batch(tokenizer, smile_strings):
    emb = tokenizer.encode_batch(smile_strings)
    assert len(emb) == 2
    assert len(emb[0].ids) == 28
    assert len(emb[1].ids) == 53  # `@@` is one token
    assert len(emb[0].ids) == len(emb[0].attention_mask)
    assert len(emb[0].ids) == len(emb[0].type_ids)
    smile_out = tokenizer.decode_batch([e.ids for e in emb])
    assert len(smile_out) == len(smile_strings)
    assert smile_out == smile_strings
    assert smile_out[0] == smile_strings[0]
    assert smile_out[1] == smile_strings[1]
    print(f"{smile_out[0]} => {emb[0].ids}")


def test_serialize(tokenizer):
    tok = deepcopy(tokenizer)
    config = json.loads(tok.to_str())
    assert config["decoder"] == {"type": "Fuse"}
    assert config["model"]["type"] == "WordLevel"
    assert config["pre_tokenizer"] == {"type": "SmirkPreTokenizer"}


def test_padding():
    tokenizer = smirk.SmirkTokenizer(padding=False)
    assert tokenizer.pad_token == ""
    tokenizer = smirk.SmirkTokenizer(padding=True)
    assert tokenizer.pad_token == "[PAD]"


def test_collate(tokenizer, smile_strings):
    collate = DataCollatorForLanguageModeling(tokenizer)
    embed = tokenizer.encode_batch(smile_strings)
    assert len(embed) == len(smile_strings)
    # assert isinstance(embed[0], Mapping)
    collated_batch = collate(embed)
