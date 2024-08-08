import json
import itertools
from pathlib import Path

import pytest
from test_fast_tokenizer import check_save, check_tokenize, check_unknown
from test_tokenize_smiles import smile_strings

import smirk

SMILE_TEST_FILE = Path(__file__).parent.joinpath("smiles.txt")


@pytest.fixture(
    scope="session",
    params=[
        {"merge_brackets": False, "split_structure": False},
        {"merge_brackets": True, "split_structure": True},
        {"merge_brackets": True, "split_structure": False},
        {"merge_brackets": False, "split_structure": True},
    ],
)
def trained(request):
    tokenizer = smirk.SmirkTokenizerFast()
    return tokenizer.train([str(SMILE_TEST_FILE)], **request.param)


def test_save(trained):
    check_save(trained)


def test_train_smirk_piece(trained, smile_strings):
    code = trained(smile_strings)
    decode = trained.batch_decode(code["input_ids"])
    assert decode == smile_strings
    assert trained.vocab_size > smirk.SmirkTokenizerFast().vocab_size
    assert "[PAD]" not in trained._tokenizer.get_vocab(with_added_tokens=False)
    assert "[PAD]" in trained.get_vocab()


def test_vocab_size():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)], vocab_size=200)
    assert trained.vocab_size == 200
    assert trained._tokenizer.get_vocab_size(False) == trained.vocab_size
    assert len(trained) == trained.vocab_size + len(smirk.SPECIAL_TOKENS) - 1
    assert (
        trained._tokenizer.get_vocab_size(True)
        - trained._tokenizer.get_vocab_size(False)
    ) == len(smirk.SPECIAL_TOKENS) - 1  # unk should already be in the vocab


def test_multi_file():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train(
        [
            str(SMILE_TEST_FILE),
            str(Path(__file__).parent.joinpath("opensmiles.smi")),
        ],
        vocab_size=215,
    )
    assert trained.vocab_size == 215


def test_tokenizing_unknown(trained):
    config = json.loads(trained.to_str())
    merges = config["model"]["merges"]
    check_unknown(trained)


def test_tokenize(trained):
    check_tokenize(trained)


@pytest.mark.parametrize("merge_brackets", [True, False])
def test_merge_brackets(merge_brackets):
    tok = smirk.SmirkTokenizerFast().train(
        [str(SMILE_TEST_FILE)],
        merge_brackets=merge_brackets,
        vocab_size=16384,
    )
    config = json.loads(tok.to_str())
    vocab = config["model"]["vocab"]
    merges = config["model"]["merges"]
    left_bracket = vocab["["]
    right_bracket = vocab["]"]
    assert left_bracket == tok.get_vocab()["["]
    assert right_bracket == tok.get_vocab()["]"]

    # Check that the right merges happened
    if merge_brackets:
        assert left_bracket in itertools.chain(*merges)
        assert right_bracket in itertools.chain(*merges)

    else:
        assert left_bracket not in itertools.chain(*merges)
        assert right_bracket not in itertools.chain(*merges)
