from smirk.smirk import SmirkTokenizer
import pytest
import json
from copy import deepcopy
from importlib.resources import files


@pytest.fixture
def tokenizer():
    VOCAB_FILE = files("smirk").joinpath("vocab_smiles.json")
    assert VOCAB_FILE.is_file()
    return SmirkTokenizer(str(VOCAB_FILE), is_smiles=True)


@pytest.fixture
def smile_strings():
    return [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
    ]


def check_encode(tok, x):
    img = tok.decode(tok.encode(x)["input_ids"])
    assert img == x


def test_pretokenize(tokenizer):
    splits = tokenizer.pretokenize("OC[C@@H][OH]")
    assert splits == ["O", "C", "[", "C", "@@", "H", "]", "[", "O", "H", "]"]
    assert len(splits) == 11


def test_image(tokenizer, smile_strings):
    check_encode(tokenizer, "OC[C@@H][OH]")
    assert len(tokenizer.encode("OC[C@@H][OH]")["input_ids"]) == 11
    for x in smile_strings:
        check_encode(tokenizer, x)


def test_encode(tokenizer, smile_strings):
    smile = "COCCC(=O)N1CCN(C)C(C2=CN([C@@H](C)C3=CC=C(C(F)(F)F)C=C3)N=N2)C1"
    emb = tokenizer.encode(smile)
    smile_out = tokenizer.decode(emb["input_ids"])
    assert smile_out == smile


def test_encode_batch(tokenizer, smile_strings):
    emb = tokenizer.encode_batch(smile_strings)
    assert len(emb) == 2
    assert len(emb[0]["input_ids"]) == len(emb[0]["attention_mask"])
    assert len(emb[0]["input_ids"]) == len(emb[0]["token_type_ids"])
    assert len(emb[0]["input_ids"]) == len(emb[0]["special_tokens_mask"])
    smile_out = tokenizer.decode_batch([e["input_ids"] for e in emb])
    assert len(smile_out) == len(smile_strings)
    assert smile_out == smile_strings
    assert smile_out[0] == smile_strings[0]
    assert smile_out[1] == smile_strings[1]


def test_serialize(tokenizer):
    tok = deepcopy(tokenizer)
    config = json.loads(tok.to_str())
    assert config["decoder"] == {"type": "Fuse"}
    assert config["model"]["type"] == "WordLevel"
    assert config["pre_tokenizer"] == {"type": "SmirkPreTokenizer"}


def test_special(tokenizer):
    assert tokenizer.special_tokens["bos_token"] == "[BOS]"
    assert tokenizer.special_tokens["eos_token"] == "[EOS]"
    assert tokenizer.special_tokens["pad_token"] == "[PAD]"
    assert tokenizer.special_tokens["sep_token"] == "[SEP]"
    assert tokenizer.special_tokens["cls_token"] == "[CLS]"
    assert tokenizer.special_tokens["mask_token"] == "[MASK]"
    assert tokenizer.special_tokens["unk_token"] == "[UNK]"
