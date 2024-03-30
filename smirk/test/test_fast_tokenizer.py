import smirk
from transformers import BatchEncoding
from transformers.data import DataCollatorForLanguageModeling
from test_tokenize_smiles import smile_strings


def test_special_tokens():
    tokenizer = smirk.SmirkTokenizerFast()
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.pad_token_id == tokenizer.get_vocab()["[PAD]"]


def test_pad(smile_strings):
    tokenizer = smirk.SmirkTokenizerFast()
    code = tokenizer(smile_strings)
    assert len(code["input_ids"][0]) != len(code["input_ids"][1])
    code = tokenizer.pad(code)
    assert len(code["input_ids"][0]) == len(code["input_ids"][1])
    assert len(code["special_tokens_mask"][0]) == len(code["special_tokens_mask"][1])


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
