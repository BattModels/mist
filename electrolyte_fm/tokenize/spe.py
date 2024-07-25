import codecs
import json
from pathlib import Path
from typing import List, Union

from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding


class PreTrainedSPETokenizer(PreTrainedTokenizerBase):
    def __init__(self, vocab_file: str | Path, spe_file: str | Path, **kwargs):
        with open(vocab_file, "r") as fid:
            self._vocab: dict[str, int] = json.load(fid)

        self._ids_to_vocab = {id: token for token, id in self._vocab.items()}

        with codecs.open(str(spe_file), "r") as fid:
            self._tokenizer = SPE_Tokenizer(fid)
        super().__init__(**kwargs)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    @property
    def vocab_size(self):
        return len(self)

    def __len__(self) -> int:
        return len(self._vocab)

    def tokenize(self, smile):
        return [
            self._convert_token_to_id(tok)
            for tok in self._tokenizer.tokenize(smile).split(" ")
        ]

    def _convert_token_to_id(self, token: str):
        vocab = self.get_vocab()
        try:
            return vocab[token]
        except KeyError:
            return vocab[self.unk_token]

    def convert_tokens_to_ids(self, tokens: str | list[str]):
        if isinstance(tokens, list):
            return [self._convert_token_to_id(token) for token in tokens]

        return self._convert_token_to_id(tokens)

    def _convert_id_to_token(self, token: int):
        return self._ids_to_vocab.get(token, self.unk_token)

    def _batch_encode_plus(
        self, batch_text_or_text_pairs: list[str], **kwargs
    ) -> BatchEncoding:
        encoding = [self.tokenize(x) for x in batch_text_or_text_pairs]
        return BatchEncoding(
            data={"input_ids": encoding},
            n_sequences=len(encoding),
        )

    def _encode_plus(self, text: str, **kwargs):
        return BatchEncoding(
            data={"input_ids": self.tokenize(text)},
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
    for id, token in enumerate(list(vocab)):
        token_to_id[token] = id

    return token_to_id


def pretrained_spe_tokenizer(cache=".cache", cache_generated=False):
    import json
    import pickle
    import shutil
    import urllib.request
    from pathlib import Path
    from zipfile import ZipFile

    # Download ChEMBL_1M_SPE: https://github.com/XinhaoLi74/MolPMoFiT
    # DOI: https://doi.org/10.6084/m9.figshare.20696935.v1
    cache = Path(cache)
    cache.mkdir(exist_ok=True)
    pretrained = cache.joinpath("spe_pretrained.zip")
    if not pretrained.is_file():
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/36910486",
            str(pretrained),
        )

    # Extract Vocab
    spe_file = cache.joinpath("SPE_ChEMBL.txt")
    if not spe_file.is_file():
        with ZipFile(pretrained) as zip:
            path = zip.extract(
                "models/SPE_ChEMBL.txt",
                path=cache,
            )
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
