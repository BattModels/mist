import smirk
import pytest
import re
from collections.abc import Mapping
from transformers.data import DataCollatorForLanguageModeling
from test_tokenize import smile_strings


def test_special_tokens():
    tokenizer = smirk.SmirkTokenizerFast()
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.pad_token_id == tokenizer.get_vocab()["[PAD]"]


def test_pad(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    embed = tokenizer(smile_strings)
    assert len(embed[0]["input_ids"]) != len(embed[1]["input_ids"])
    embed = tokenizer.pad(embed)
    assert len(embed["input_ids"][0]) == len(embed["input_ids"][1])
    assert len(embed["special_tokens_mask"][0]) == len(embed["special_tokens_mask"][1])


def test_collate(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    collate = DataCollatorForLanguageModeling(tokenizer)
    embed = tokenizer(smile_strings)
    assert len(embed) == len(smile_strings)
    assert isinstance(embed[0], Mapping)
    assert "special_tokens_mask" in embed[0].keys()

    # Collate batch
    collated_batch = collate(embed)
    print(collated_batch)

    # Should pad to longest
    for k in [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    ]:
        assert k in collated_batch.keys()
        assert collated_batch[k].size() == (2, len(embed[1]["input_ids"]))

    # Check for padding
    n = len(embed[0]["input_ids"]) - 1
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
