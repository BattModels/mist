# Import Rust Binding
import os
from importlib.resources import files
from typing import List, Optional, Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import AddedToken, SpecialTokensMixin
from transformers.tokenization_utils_fast import TOKENIZER_FILE
from transformers.utils.generic import PaddingStrategy

from . import smirk as rs_smirk

SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}


class SmirkTokenizerFast(PreTrainedTokenizerBase, SpecialTokensMixin):
    def __init__(self, **kwargs):
        # Create SmirkTokenizer
        default_vocab_file = str(files("smirk").joinpath("vocab_smiles.json"))
        if tokenizer := kwargs.pop("tokenizer", None):
            tokenizer = tokenizer
        elif tokenizer_file := kwargs.pop("tokenizer_file", None):
            tokenizer = rs_smirk.SmirkTokenizer.from_file(tokenizer_file)
        elif vocab_file := kwargs.pop("vocab_file", default_vocab_file):
            is_smiles = kwargs.pop("is_smiles", True)
            tokenizer = rs_smirk.SmirkTokenizer.from_vocab(
                vocab_file, is_smiles=is_smiles
            )
        else:
            tokenizer = rs_smirk.SmirkTokenizer()

        self._tokenizer = tokenizer
        self.verbose = kwargs.pop("verbose", False)
        super().__init__(**kwargs)

        if kwargs.pop("add_special_tokens", True):
            self.add_special_tokens(SPECIAL_TOKENS)

    def __len__(self) -> int:
        """Size of the full vocab with added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def __repr__(self):
        return self.__class__.__name__

    def is_fast(self):
        return True

    def to_str(self) -> str:
        return self._tokenizer.to_str()

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        return {
            id: AddedToken(content)
            for id, content in self._tokenizer.get_added_tokens_decoder().items()
        }

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        return {
            content: id
            for id, content in self._tokenizer.get_added_tokens_decoder().items()
        }

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        # Normalize to AddedTokens
        new_tokens = [
            (
                AddedToken(token, special=special_tokens)
                if isinstance(token, str)
                else token
            )
            for token in new_tokens
        ]
        return self._tokenizer.add_tokens(new_tokens)

    def batch_decode_plus(self, ids, **kwargs):
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        return self._tokenizer.decode_batch(ids)(
            ids, skip_special_tokens=skip_special_tokens
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        add_special_tokens = kwargs.pop("add_special_tokens", True)
        encoding = self._tokenizer.encode_batch(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )
        batch = BatchEncoding(
            data={k: [dic[k] for dic in encoding] for k in encoding[0]},
            encoding=encoding,
            n_sequences=len(encoding),
        )
        if kwargs.pop("padding_strategy") is not PaddingStrategy.DO_NOT_PAD:
            return pad_without_fast_tokenizer_warning(
                self, batch, return_tensors=kwargs.pop("return_tensors", "pt"), **kwargs
            )
        return batch

    def _encode_plus(
        self, text: str, add_special_tokens: bool = True, **kwargs
    ) -> BatchEncoding:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return BatchEncoding(data=encoding, encoding=encoding, n_sequences=1)

    def _decode(self, token_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def tokenize(self, text: str, add_special_tokens=False) -> list[str]:
        """Converts a string into a sequence of tokens"""
        return self._tokenizer.tokenize(text, add_special_tokens)

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab_size(self):
        """The size of the vocabulary without the added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def convert_tokens_to_ids(
        self, tokens: Union[str, list[str]]
    ) -> Union[int, list[int]]:
        vocab = self.get_vocab()
        if isinstance(tokens, str):
            return vocab[tokens]
        return [vocab[token] for token in tokens]

    def _save_pretrained(
        self,
        save_directory,
        file_names,
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> tuple[str]:
        assert legacy_format is None or not legacy_format
        tokenizer_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE,
        )
        self._tokenizer.save(tokenizer_file)
        return file_names + (tokenizer_file,)

    def train(self, files: list[str], **kwargs) -> "SmirkTokenizerFast":
        """Train a SmirkPiece Model from files

        files: List of files containing the corpus to train the tokenizer on
        min_frequency: Minimum count for a pair to be considered for a merge
        vocab_size: the target size of the final vocabulary
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return SmirkTokenizerFast(tokenizer=self._tokenizer.train(files, **kwargs))
