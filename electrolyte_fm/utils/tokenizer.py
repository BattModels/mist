from transformers import PreTrainedTokenizerBase


def load_tokenizer(name, **kwargs) -> PreTrainedTokenizerBase:
    # Locate Tokeniser and dataset
    unk_name = RuntimeError(f"Unknown tokenizer: {name}")
    if name.startswith("smirk"):
        from smirk import SmirkTokenizerFast

        if name == "smirk":
            return SmirkTokenizerFast(is_smiles=True)
        elif name == "smirk-selfies":
            return SmirkTokenizerFast(is_smiles=False)

        raise unk_name

    elif name == "SmilesPE/SPE_ChEMBL":
        from ..tokenize.spe import pretrained_spe_tokenizer

        return pretrained_spe_tokenizer()

    else:
        # Fall back to a HuggingFace Tokenizer
        from transformers import AutoTokenizer
        from smirk import SmirkTokenizerFast

        AutoTokenizer.register(
            "SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast
        )

        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
            **kwargs,
        )
