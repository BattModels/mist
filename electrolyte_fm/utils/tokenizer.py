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

    elif (
        Path(name).is_dir()
        and Path(name).parent.parent.joinpath("config.json").exists()
    ):
        # Reload Tokenizer from Checkpoint
        from ..utils.ckpt import get_ckpt_tokenizer

        tokenizer = get_ckpt_tokenizer(name)
        print(f"Loading {tokenizer} for {name}")
        return load_tokenizer(tokenizer, **kwargs)

    elif name == "ibm/MoLFormer-XL-both-10pct-oov":
        # By default "ibm/MoLFormer-XL-both-10pct" strips out unknown tokens
        # during the pre-tokenization step, this reverts that to ensure
        # unknown tokens are emitted
        import json
        from transformers import AutoTokenizer
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import Split

        tok_tf = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            cache=".cache",
            **kwargs,
        )
        config = json.loads(tok_tf.backend_tokenizer.to_str())
        regex = config["pre_tokenizer"]["pretokenizers"][-1]["pattern"]["Regex"]
        tok_tf.backend_tokenizer.pre_tokenizer = Split(
            Regex(regex), "isolated")
        return tok_tf

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
