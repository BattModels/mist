from importlib.resources import files
from pathlib import Path

import pytest
import selfies
from elements import all_bracketed_tokens, all_elements

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


def test_elements(tokenizer):
    def check_encode(b, n):
        b = selfies.encoder(b, strict=True)
        code = tokenizer.pretokenize(b)
        assert len(code) == n

    for element in all_elements():
        check_encode(f"[{element}]", 3)
        check_encode(f"[{element}@]", 4)
        check_encode(f"[{element}@@]", 4)
        check_encode(f"[{element}+2]", 5)


def test_bracketed_tokens(tokenizer):
    unk_token_id = tokenizer.get_vocab()["[UNK]"]
    for b in all_bracketed_tokens(include_aromatic=False):
        emb = tokenizer.encode(b)
        assert (
            unk_token_id not in emb["input_ids"]
        ), f"failed to tokenize: {b} => {tokenizer.pretokenize(b)} ({emb})"


def test_opensmile_spec(tokenizer):
    with open(Path(__file__).parent.join("opensmiles.smi"), "r") as examples:
        unk_token_id = tokenizer.token_to_id("[UNK]")
        for smile in examples:
            # Skip comments
            if smile.startswith("#"):
                continue

            emb = tokenizer.encode(smile)
            assert (
                unk_token_id not in emb["input_ids"]
            ), f"failed to tokenize: {smile} => {tokenizer.pretokenize(smile)} ({emb})"
