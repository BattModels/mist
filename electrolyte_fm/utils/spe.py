import codecs
import json
from pathlib import Path
from typing import List, Union
import pickle
import shutil
from zipfile import ZipFile

from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from .cache import cached_download


class PreTrainedSPETokenizer(PreTrainedTokenizerBase):
    def __init__(self, vocab_file: str | Path, spe_file: str | Path, **kwargs):
        with open(vocab_file, "r") as fid:
            self._vocab: dict[str, int] = json.load(fid)

        self._ids_to_vocab = {id: token for token, id in self._vocab.items()}

        with codecs.open(str(spe_file), "r") as fid:
            self._tokenizer = SPE_Tokenizer(fid)

        self._merges_file = Path(spe_file)

        super().__init__(**kwargs)

    def __repr__(self):
        return "SPETokenizer"

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    @property
    def vocab_size(self):
        return len(self)

    def is_fast(self):
        return False

    def __len__(self) -> int:
        return len(self._vocab)

    def _flush_cache(self):
        # SPE_Tokenizer maintains a cache to speed up tokenization, but it
        # doesn't have limits on it's size, causing memory issues. This
        # function flushes the cache to prevent memory issues
        self._tokenizer.cache.clear()

    def tokenize(self, smile: str) -> list[str]:
        code = self(smile)["input_ids"]
        self._flush_cache()
        return [self._convert_id_to_token(id) for id in code]

    def _convert_token_to_id(self, token: str) -> int:
        vocab = self.get_vocab()
        try:
            return vocab[token]
        except KeyError:
            return self.unk_token_id

    def convert_tokens_to_ids(self, tokens: str | list[str]):
        if isinstance(tokens, list):
            return [self._convert_token_to_id(token) for token in tokens]

        return self._convert_token_to_id(tokens)

    def _convert_id_to_token(self, token: int):
        return self._ids_to_vocab.get(token, self.unk_token)

    def convert_ids_to_tokens(self, tokens: int | list[int]):
        if isinstance(tokens, int):
            return self._convert_id_to_token(tokens)
        return [self._convert_id_to_token(token) for token in tokens]

    def _batch_encode_plus(
        self, batch_text_or_text_pairs: list[str], **kwargs
    ) -> BatchEncoding:
        encoding = [self._encode_plus(x) for x in batch_text_or_text_pairs]
        self._flush_cache()
        return BatchEncoding(
            data={"input_ids": [x["input_ids"] for x in encoding]},
            n_sequences=len(encoding),
        )

    def _encode_plus(self, text: str, **kwargs):
        input_ids = [
            self._convert_token_to_id(token)
            for token in self._tokenizer.tokenize(text).split(" ")
        ]
        self._flush_cache()
        return BatchEncoding(
            data={"input_ids": input_ids},
            n_sequences=1,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        token_ids = [token_ids] if isinstance(token_ids, int) else token_ids
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        return "".join(tokens)

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(exist_ok=True, parents=True)
        Path(save_directory, "vocab.json").write_text(json.dumps(self._vocab))
        Path(save_directory, "spe_merges.txt").write_text(self._spe_file.read_text())


def process_vocab(vocab_list: list[str]) -> dict[str, int]:
    vocab = set(vocab_list)
    vocab -= set(["xxfake"])
    vocab |= set(
        [
            "[UNK]",
            "[MASK]",
            "[BOS]",
            "[EOS]",
            "[CLS]",
            "[SEP]",
        ]
    )

    token_to_id = {}
    vocab = list(vocab)
    vocab.sort()
    for id, token in enumerate(vocab):
        token_to_id[token] = id

    return token_to_id


def pretrained_spe_tokenizer(cache_generated=False):
    # Download ChEMBL_1M_SPE: https://github.com/XinhaoLi74/MolPMoFiT
    # DOI: https://doi.org/10.6084/m9.figshare.20696935.v1
    pretrained = cached_download(
        "https://figshare.com/ndownloader/files/36910486", "spe_pretrained.zip"
    )

    # Extract Vocab
    cache = pretrained.parent
    spe_file = cache.joinpath("SPE_ChEMBL.txt")
    if not spe_file.is_file():
        with ZipFile(pretrained) as zip:
            path = zip.extract("models/SPE_ChEMBL.txt", cache)
            shutil.move(path, spe_file)

    vocab_file = cache.joinpath("ChEMBL_LM_SPE_vocab.json")
    if not vocab_file.is_file() or not cache_generated:
        with ZipFile(pretrained) as zip:
            with zip.open("models/ChEMBL_LM_SPE_vocab.pkl", "r") as fid:
                vocab_list = pickle.load(fid)

        vocab = process_vocab(vocab_list)
        with open(vocab_file, "w") as fid:
            json.dump(vocab, fid)

    return PreTrainedSPETokenizer(
        vocab_file=vocab_file,
        spe_file=spe_file,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
    )
