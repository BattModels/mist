import pytest
from tempfile import TemporaryDirectory
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import smirk
from electrolyte_fm.tokenize.spe import PreTrainedSPETokenizer, pretrained_spe_tokenizer
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


@pytest.fixture(scope="module", params=SMILE_TOKENIZER)
def smile_tokenizer(request):
    return load_tokenizer(request.param)


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
    if isinstance(smile_tokenizer, PreTrainedSPETokenizer):
        pytest.xfail("Out of vocab for SPETokenizer")
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


def test_spe_setup():
    tokenizer = pretrained_spe_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    vocab = tokenizer.get_vocab()
    assert "xxfake" not in vocab
    assert "[BOS]" in vocab
    assert "[N+]" in vocab
