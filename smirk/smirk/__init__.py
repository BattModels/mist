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
        default_vocab_file = files("smirk").joinpath("vocab_smiles.json")
        if tokenizer := kwargs.pop("tokenizer", None):
            tokenizer = tokenizer
        elif tokenizer_file := kwargs.pop("tokenizer_file", None):
            tokenizer = rs_smirk.SmirkTokenizer.from_file(str(tokenizer_file))
        elif vocab_file := kwargs.pop("vocab_file", default_vocab_file):
            tokenizer = rs_smirk.SmirkTokenizer.from_vocab(str(vocab_file))
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

    def batch_decode_plus(self, ids, **kwargs) -> list[str]:
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        return self._tokenizer.decode_batch(
            ids, skip_special_tokens=skip_special_tokens
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs) -> BatchEncoding:
        add_special_tokens = kwargs.pop("add_special_tokens", True)
        assert not kwargs.pop(
            "is_pretokenized", False
        ), "Pretokenized input is not supported"
        encoding = self._tokenizer.encode_batch(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )

        # Convert encoding to dict
        data = {
            "input_ids": [x["input_ids"] for x in encoding],
            "token_type_ids": [x["token_type_ids"] for x in encoding],
            "attention_mask": [x["attention_mask"] for x in encoding],
            "special_tokens_mask": [x["special_tokens_mask"] for x in encoding],
        }
        if kwargs.pop("return_offsets_mapping", False):
            data["offset_mapping"] = [x["offsets"] for x in encoding]

        batch = BatchEncoding(
            data,
            encoding,
            n_sequences=len(encoding),
            tensor_type=kwargs.get("return_tensors", None),
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
        data = {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding["token_type_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
        }
        if kwargs.pop("return_offsets_mapping", False):
            data["offset_mapping"] = encoding["offsets"]

        return BatchEncoding(data, encoding, n_sequences=1)

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


def SmirkSelfiesFast(vocab=None, unk_token="[UNK]", add_special_tokens=True, **kwargs):
    """Instantiate a Chemically-Consistent tokenizer for SELFIES

    Defaults to a vocab of all possible SELFIES tokens plus the `[UNK]` for
    the unknown token. Additional kwargs are passed to `PreTrainedTokenizerFast`
    """
    import json
    from importlib.resources import files

    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import Strip
    from tokenizers.pre_tokenizers import Sequence, Split
    from transformers import PreTrainedTokenizerFast

    if vocab is None:
        with open(files("smirk").joinpath("vocab_selfies.json"), "r") as fid:
            vocab = json.load(fid)

    tok = Tokenizer(WordLevel(vocab, unk_token))
    # Regex generated using `opt/build_vocab.py -f smiles -t regex`
    regex = (
        r"Branch|Ring|A[c|g|l|m|r|s|t|u]|B[a|e|h|i|k|r]?|"
        r"C[a|d|e|f|l|m|n|o|r|s|u]?|D[b|s|y]|E[r|s|u]|F[e|l|m|r]?|"
        r"G[a|d|e]|H[e|f|g|o|s]?|I[n|r]?|Kr?|L[a|i|r|u|v]|"
        r"M[c|d|g|n|o|t]|N[a|b|d|e|h|i|o|p]?|O[g|s]?|P[a|b|d|m|o|r|t|u]?|"
        r"R[a|b|e|f|g|h|n|u]|S[b|c|e|g|i|m|n|r]?|T[a|b|c|e|h|i|l|m|s]|"
        r"U|V|W|Xe|Yb?|Z[n|r]|[\.\-=\#\$:/\\\+\-]|\d|@|@@"
    )

    tok.pre_tokenizer = Sequence(
        [
            Split(Regex(r"\[|]"), behavior="removed"),  # Strip Brackets
            Split(Regex(regex), behavior="isolated"),  # Tokenize
        ]
    )
    tok.normalizer = Strip()
    tok_tf = PreTrainedTokenizerFast(tokenizer_object=tok, **kwargs)

    if kwargs.pop("add_special_tokens", True):
        tok_tf.add_special_tokens(SPECIAL_TOKENS)

    return tok_tf
