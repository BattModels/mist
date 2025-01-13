import json
import urllib
from tempfile import TemporaryDirectory
from itertools import chain

import pytest
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import smirk
from electrolyte_fm.utils.tokenizer import load_tokenizer

SMILE_TOKENIZER = [
    "smirk",
    "SmilesPE/SPE_ChEMBL",
    "ibm/MoLFormer-XL-both-10pct",
]

STANDARD_SMILES = [
    "CC[N+](C)(C)Cc1ccccc1Br",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN(CCC1=CNC2=C1C=CC=C2)C",
    "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
]

# Tokenizers not actively used for training
OTHER_SMILES_TOKENIZERS = [
    "character",
    "SmilesPE/SPE_ChEMBL",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "devalab/molgpt-moses",
    "MolecularAI/Chemformer",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
]


def flaky_load_tokenizer(name_or_path, *args, **kwargs):
    try:
        return load_tokenizer(name_or_path, *args, **kwargs)

    except (json.decoder.JSONDecodeError, urllib.error.HTTPError):
        if name_or_path in ["SmilesPE/SPE_ChEMBL"]:
            pytest.xfail("SPE tokenizer not available (flaky download)")
        raise

    except ImportError:
        pytest.xfail(f"Dependency not installed for {name_or_path}")


@pytest.fixture(scope="module", params=SMILE_TOKENIZER)
def smile_tokenizer(request):
    return flaky_load_tokenizer(request.param)


@pytest.mark.parametrize("name", chain(SMILE_TOKENIZER, OTHER_SMILES_TOKENIZERS))
def test_well_behaved_tokenizer(name):
    tokenizer = flaky_load_tokenizer(name)
    code = tokenizer("CCO")
    tokens = tokenizer.tokenize("CCO")
    assert isinstance(tokens, list) and len(tokens) >= 1 and isinstance(tokens[0], str)
    assert tokenizer.mask_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id not in code["input_ids"]

    for special in [tokenizer.mask_token, tokenizer.pad_token, tokenizer.unk_token]:
        assert special in tokenizer.all_special_tokens
        assert tokenizer.encode(special)[0] in tokenizer.all_special_ids

    vocab = tokenizer.get_vocab()
    assert vocab[tokenizer.mask_token] == tokenizer.mask_token_id

    check_encoding(tokenizer, ["CCO", "C-C-O", "CC(C)C(=O)C(C)C"])


@pytest.mark.parametrize("name", chain(SMILE_TOKENIZER, OTHER_SMILES_TOKENIZERS))
def test_oov_tokens(name):
    """Check that the unknown token is emitted for OOV tokens"""
    tok = flaky_load_tokenizer(name)

    def check_oov(tokenizer, smi):
        code = tokenizer(smi)["input_ids"]
        assert tok.unk_token_id in code
        assert tok.decode(code, skip_special_tokens=False) != smi
        assert str(tok.unk_token) in tok.decode(code)

    # Some Tokenizers don't emit the unknown token no matter what the input is.
    # Check that the tokenizer fails the test, then mark it with xfail
    if name == "ibm/MoLFormer-XL-both-10pct":
        assert tok.unk_token_id not in tok("😬")["input_ids"]
        pytest.xfail("MoLFormer strips unknown tokens pre-tokenizer")
    elif name in ["seyonec/ChemBERTa-zinc-base-v1", "ChangwenXu98/TransPolymer"]:
        assert tok.unk_token_id not in tok("⛰️ ⋙ 🏖️")["input_ids"]
        pytest.xfail("open vocab model")
    elif name == "character":
        check_oov(tok, "⚛️")
        assert tok.unk_token_id not in tok("ZZ[Zz]")["input_ids"]
        pytest.xfail("character-level tokenizer")

    check_oov(tok, "Zz")
    check_oov(tok, "[Zz]")
    check_oov(tok, "[Zz&3]")


def test_vocab_size(smile_tokenizer):
    # vocab size (Size of model vocab without added tokens)
    # should be smaller (or equal if the model knows of all added tokens)
    # than the length of of the tokenizer
    assert smile_tokenizer.vocab_size <= len(smile_tokenizer)


def test_pretrained_smirk():
    tok = smirk.SmirkTokenizerFast()

    with TemporaryDirectory() as dir:
        tok.save_pretrained(dir)
        loaded = load_tokenizer(str(dir))
        assert isinstance(loaded, smirk.SmirkTokenizerFast)
        assert tok.to_str() == loaded.to_str()


def check_encoding(tokenizer: PreTrainedTokenizerBase, batch: list[str]):
    # Check output
    code = tokenizer(batch)
    assert isinstance(code, BatchEncoding)
    assert "input_ids" in code.keys()
    assert isinstance(code["input_ids"], list)
    assert isinstance(code["input_ids"][0], list)
    assert isinstance(code["input_ids"][0][0], int)

    # Check reverse
    assert len(code["input_ids"]) == len(batch)
    out = tokenizer.batch_decode(code["input_ids"], skip_special_tokens=True)
    assert len(out) == len(batch)
    assert out == batch


def test_standard_smiles(smile_tokenizer):
    check_encoding(smile_tokenizer, STANDARD_SMILES)


def test_compounds_smiles(smile_tokenizer):
    check_encoding(
        smile_tokenizer,
        [
            "[Sc+3].[OH-].[OH-].[OH-]",
            "[Li+].F[P-](F)(F)(F)(F)F",
        ],
    )


def test_mlm_tokenizer(smile_tokenizer):
    collate = DataCollatorForLanguageModeling(smile_tokenizer, mlm_probability=0.5)
    code = [smile_tokenizer(smile) for smile in STANDARD_SMILES]
    print(code)
    collated_batch = collate(code)
    print(collated_batch)

    # Should pad to longest
    max_length = max((len(x["input_ids"]) for x in code))
    for k in [
        "input_ids",
        # "token_type_ids",
        "attention_mask",
        "labels",
    ]:
        assert k in collated_batch.keys()
        assert collated_batch[k].size() == (len(STANDARD_SMILES), max_length)


@pytest.mark.xfail(
    strict=False, raises=urllib.error.HTTPError, reason="Flaky downloads"
)
def test_spe_setup():
    try:
        from electrolyte_fm.tokenize.spe import (
            PreTrainedSPETokenizer,
            pretrained_spe_tokenizer,
        )
    except ImportError:
        pytest.skip("SmilesPE not installed")

    tokenizer = pretrained_spe_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    vocab = tokenizer.get_vocab()
    assert "xxfake" not in vocab
    assert "[BOS]" in vocab
    assert "[N+]" in vocab
    assert "[N+]" in vocab
