import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from test_tokenize_smiles import smile_strings
from transformers import BatchEncoding
from transformers.data import DataCollatorForLanguageModeling

import smirk


def check_save(tokenizer):
    """Check that the tokenzier can be saved"""
    with TemporaryDirectory() as save_directory:
        files = tokenizer.save_pretrained(save_directory)
        assert len(files) == 3
        for file in files:
            file = Path(file)
            assert file.is_file()
            assert file.suffix == ".json"

        loaded = tokenizer.from_pretrained(save_directory)
    assert isinstance(loaded, tokenizer.__class__)
    assert loaded.to_str() == tokenizer.to_str()

    # Check pickling
    state = tokenizer.__getstate__()
    pickled = tokenizer.__class__.__setstate__(state)
    assert pickled.to_str() == tokenizer.to_str()


def test_saving():
    tokenizer = smirk.SmirkTokenizerFast()
    check_save(tokenizer)


def test_vocab_size():
    tokenizer = smirk.SmirkTokenizerFast()
    assert len(tokenizer.get_vocab()) > 0
    assert len(tokenizer.get_vocab()) == tokenizer.vocab_size


def test_special_tokens():
    tokenizer = smirk.SmirkTokenizerFast()
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.mask_token_id == 144
    assert tokenizer.pad_token_id == 145
    assert tokenizer.unk_token_id == 147


def test_pad(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    code = tokenizer(smile_strings)
    assert len(code["input_ids"][0]) != len(code["input_ids"][1])
    code = tokenizer.pad(code)
    assert len(code["input_ids"][0]) == len(code["input_ids"][1])
    assert len(code["special_tokens_mask"][0]) == len(code["special_tokens_mask"][1])


def test_encode(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    codes = [tokenizer(smile) for smile in smile_strings]
    for code, smile in zip(codes, smile_strings):
        assert "input_ids" in code
        assert "special_tokens_mask" in code
        assert "attention_mask" in code
        assert tokenizer.decode(code["input_ids"], skip_special_tokens=True) == smile


@pytest.mark.xfail(reason="smirk skips unknown tokens")
def test_unk(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    code = tokenizer("🤷")["input_ids"]
    assert code == [tokenizer.unk_token_id]
    code = tokenizer(["🤷"])["input_ids"][0]
    assert code == [tokenizer.unk_token_id]


def test_collate(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    collate = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.5)
    code = tokenizer(smile_strings)
    assert len(code["input_ids"]) == len(smile_strings)
    assert isinstance(code, BatchEncoding)
    assert "special_tokens_mask" in code.keys()

    # Collate batch
    collated_batch = collate(code)
    print(collated_batch)

    # Should pad to longest
    max_length = len(code["input_ids"][1])
    for k in [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    ]:
        assert k in collated_batch.keys()
        assert collated_batch[k].size() == (2, max_length)

    # Check for padding
    n = len(code["input_ids"][0]) - 1
    collated_batch["input_ids"][0, n:] == tokenizer.pad_token_id

    # Check decode
    decode = tokenizer.batch_decode(collated_batch["input_ids"])
    assert tokenizer.pad_token in decode[0]
    assert tokenizer.mask_token in decode[0]
    assert tokenizer.mask_token in decode[1]
    decode_no_special = tokenizer.batch_decode(
        collated_batch["input_ids"], skip_special_tokens=True
    )
    assert len(decode_no_special[0]) < len(decode[0])
    assert tokenizer.pad_token not in decode_no_special[0]
