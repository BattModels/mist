# Import Rust Binding
from . import smirk as rs_smirk
from pathlib import Path
from transformers import PreTrainedTokenizerFast

VOCAB_FILE = Path(__file__).parent.parent.joinpath("vocab.json")


class SmirkTokenizer(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["ids", "attention_mask", "type_ids"]

    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        vocab_file=VOCAB_FILE,
        **kwargs,
    ):
        # check for unsupported flags
        assert (
            "chat_template" not in kwargs.keys()
        ), f"chat_template not supported by {self.__class__}"
        assert (
            "split_special_tokens" not in kwargs.keys()
        ), f"{self.__class__} doesn't support splitting special tokens"

        # Construct Tokenizer
        if "tokenizer_object" not in kwargs.keys():
            kwargs["tokenizer_object"] = rs_smirk.SmirkTokenizer(str(vocab_file))

        super().__init__(**kwargs)
