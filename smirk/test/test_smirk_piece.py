import pytest
from pathlib import Path

from test_fast_tokenizer import check_save, check_tokenize
from test_tokenize_smiles import smile_strings

import smirk

SMILE_TEST_FILE = Path(__file__).parent.joinpath("smiles.txt")


@pytest.fixture
def trained():
    tokenizer = smirk.SmirkTokenizerFast()
    return tokenizer.train([str(SMILE_TEST_FILE)])


def test_train(trained, smile_strings):
    check_save(trained)

    # Check inversion
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


def test_multi_file():
    tokenizer = smirk.SmirkTokenizerFast()
    trained = tokenizer.train([str(SMILE_TEST_FILE)], vocab_size=200)
    assert trained.vocab_size == 200


def test_unk(trained, smile_strings):
    vocab = trained._tokenizer.get_vocab(with_added_tokens=False)
    assert trained.unk_token in vocab.keys()
    code = trained("🤷")["input_ids"]
    assert code == [trained.unk_token_id]
    code = trained(["🤷"])["input_ids"][0]
    assert code == [trained.unk_token_id]


def test_tokenize(trained):
    check_tokenize(trained)
