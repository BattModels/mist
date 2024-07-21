from pathlib import Path

import pytest
from test_fast_tokenizer import check_save, check_tokenize, check_unknown
from test_tokenize_smiles import smile_strings

import smirk

SMILE_TEST_FILE = Path(__file__).parent.joinpath("smiles.txt")


@pytest.fixture
def trained():
    tokenizer = smirk.SmirkTokenizerFast()
    return tokenizer.train([str(SMILE_TEST_FILE)])


def test_save(trained):
    check_save(trained)


def test_train_smirk_piece(trained, smile_strings):
    code = trained(smile_strings)
    decode = trained.batch_decode(code["input_ids"])
    assert decode == smile_strings
    assert trained.vocab_size > smirk.SmirkTokenizerFast().vocab_size
    print(trained.get_vocab())
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
    ) == len(
        smirk.SPECIAL_TOKENS
    ) - 1  # unk should already be in the vocab


def test_multi_file():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)], vocab_size=300)
    print(trained.get_vocab())
    print(trained.to_str())
    assert len(trained) == 300


def test_tokenizing_unknown(trained):
    check_unknown(trained)


def test_tokenize(trained):
    check_tokenize(trained)
