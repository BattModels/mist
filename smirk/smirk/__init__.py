# Import Rust Binding
from . import smirk as rs_smirk
from pathlib import Path
from typing import Union
from transformers import PreTrainedTokenizerBase

VOCAB_FILE = str(Path(__file__).parent.parent.joinpath("vocab.json"))

# Expose chemically_consistent_split
from .smirk import chemically_consistent_split


class SmirkTokenizerFast(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        # Create SmirkTokenizer
        if tokenizer_file := kwargs.pop("tokenizer_file", None):
            tokenizer = rs_smirk.SmirkTokenizer.from_file(tokenizer_file)
        elif vocab_file := kwargs.pop("vocab_file", VOCAB_FILE):
            padding = kwargs.pop("padding", False)
            tokenizer = rs_smirk.SmirkTokenizer(vocab_file)
        self._tokenizer = tokenizer

        # Add special tokens
        kwargs.update(tokenizer.special_tokens)

        super().__init__(**kwargs)

    def __len__(self) -> int:
        """Size of the full vocab with added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def is_fast(self):
        return True

    def _batch_decode_plus(self, ids, **kwargs):
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        return self._tokenizer.decode_batch(ids)(
            ids, skip_special_tokens=skip_special_tokens
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        add_special_tokens = kwargs.pop("add_special_tokens", True)
        return self._tokenizer.encode_batch(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )

    def _decode(self, token_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_vocab(self):
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def convert_tokens_to_ids(
        self, tokens: Union[str, list[str]]
    ) -> Union[int, list[int]]:
        vocab = self.get_vocab()
        if isinstance(tokens, str):
            return vocab[tokens]
        return [vocab[token] for token in tokens]
