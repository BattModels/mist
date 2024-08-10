from pathlib import Path

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


def load_tokenizer(name: str, **kwargs) -> PreTrainedTokenizerBase:
    # Locate Tokeniser and dataset
    unk_name = RuntimeError(f"Unknown tokenizer: {name}")
    if name.startswith("smirk"):
        from smirk import SmirkTokenizerFast

        if name == "smirk":
            return SmirkTokenizerFast(is_smiles=True)
        elif name == "smirk-selfies":
            from smirk import SmirkSelfiesFast

            return SmirkSelfiesFast()

        raise unk_name

    elif name == "SmilesPE/SPE_ChEMBL":
        from ..tokenize.spe import pretrained_spe_tokenizer

        return pretrained_spe_tokenizer()

    elif Path(name).parent.parent.joinpath("config.json").exists():
        # Reload Tokenizer from Checkpoint
        from ..utils.ckpt import get_ckpt_tokenizer

        tokenizer = get_ckpt_tokenizer(name)
        return load_tokenizer(tokenizer, **kwargs)

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
