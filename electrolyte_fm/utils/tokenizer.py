# Additional imports should be delayed until use by the respective loaders
# This is used during training where startup time matters
from pathlib import Path

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


def load_tokenizer(name, **kwargs) -> PreTrainedTokenizerBase:
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

    elif name.startswith("rxn4chemistry"):
        if name == "rxn4chemistry/rxnfp":
            # https://github.com/rxn4chemistry/rxnfp/blob/9c8e4a2b90399a42403481db5d40a806c310e69e/rxnfp/tokenization.py#L23
            # Matches version from https://doi.org/10.5281/zenodo.4277570
            regex = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

            # Default vocab file
            # https://github.com/rxn4chemistry/rxnfp/blob/9c8e4a2b90399a42403481db5d40a806c310e69e/rxnfp/tokenization.py#L25-L32
            vocab_file = cached_github_archive(
                "rxn4chemistry/rxnfp",
                "9c8e4a2b90399a42403481db5d40a806c310e69e",
                "rxnfp/models/transformers/bert_ft_10k_25s/vocab.txt",
            )
            return regex_smiles_tokenizer(vocab_from_words(vocab_file), regex)

        elif name == "rxn4chemistry/rxn_yields":
            # Using the latest release at time of analysis (6/22/2024)
            # Manually verified that all vocab files are identical hashing to
            # `sha256:399868a653e85549ce150af4a1f6cd2776240c6a3d951ced2120565f00f027fd`
            vocab_file = cached_github_archive(
                "rxn4chemistry/rxn_yields",
                "a126bb6c7cc66f59c811e336ae117b6ad9ac85e1",
                "trained_models/uspto/uspto_milligram_time_test_epochs_2_pretrained/checkpoint-27558-epoch-2/vocab.txt",
            )

            # Using regex from rxnfp, as Yield-BERT uses tokenizer from there:
            # https://github.com/rxn4chemistry/rxn_yields/blob/a126bb6c7cc66f59c811e336ae117b6ad9ac85e1/rxn_yields/core.py#L12
            # Exact version of rxnfp was undocumented using the latest at time of analysis (6/22/2024)
            # https://github.com/rxn4chemistry/rxnfp/blob/865880e0ba27a932d77e698cbe294af0a20a5c88/rxnfp/tokenization.py#L21
            regex = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

            return regex_smiles_tokenizer(
                vocab_from_words(vocab_file, add_unk_token=False),
                regex,
            )

    elif name.startswith("MolecularAI/Chemformer"):
        # https://github.com/MolecularAI/Chemformer/blob/0ca3c1b5f810a0ff106bbb846f511629eac3b4a5/molbart/build_tokeniser.py#L13
        regex = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

        # Load Vocab
        filename = (
            "bart_vocab_downstream.txt"
            if name.endswith("downstream")
            else "bart_vocab.txt"
        )
        vocab_file = cached_github_archive(
            "MolecularAI/Chemformer",
            "0ca3c1b5f810a0ff106bbb846f511629eac3b4a5",
            filename,
        )
        return regex_smiles_tokenizer(vocab_from_words(vocab_file), regex)

    elif name.startswith("devalab/molgpt"):
        # Source: https://github.com/devalab/molgpt/blob/72ff33ae747c0a4908b822732019a66e965a595a/train/train.py#L96C15-L96C122
        regex = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

        repo_ver = ("devalab/molgpt", "72ff33ae747c0a4908b822732019a66e965a595a")
        if name.endswith("moses"):
            file = cached_github_archive(*repo_ver, "moses2_stoi.json")
        else:
            file = cached_github_archive(*repo_ver, "guacamol2_stoi.json")

        return regex_smiles_tokenizer(vocab_from_words(file), regex)

    elif name == "ChangwenXu98/TransPolymer":
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import Split
        from transformers import AutoTokenizer

        tok_tf = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        tok = tok_tf.backend_tokenizer

        # https://github.com/ChangwenXu98/TransPolymer/blob/8399d4816ce772b64deba34f4455d91d9a764b2a/PolymerSmilesTokenization.py#L225
        tok.pre_tokenizers = Split(
            Regex(
                r"(\-?[0-9]+\.?[0-9]*|\[|\]|SELF|Li|Be|Na|Mg|Al|K|Ca|Co|Zn|Ga|Ge|As|Se|Sn|Te|N|O|P|H|I|b|c|n|o|s|p|Br?|Cl?|Fe?|Ni?|Si?|\||\(|\)|\^|=|#|-|\+|\\|\/|@|\*|\.|\%|\$)"
            ),
            "isolated",
        )
        return tok_tf

    elif name == "HUBioDataLab/SELFormer":
        from tokenizers import Tokenizer

        tok_file = cached_github_archive(
            "HUBioDataLab/SELFormer",
            "86088595867c72105f499a003e97074eddcff799",
            "data/BPETokenizer/bpe.json",
        )
        tok = Tokenizer.from_file(str(tok_file))
        tok_tf = PreTrainedTokenizerFast(tokenizer_object=tok)
        ensure_special_tokens(tok_tf)
        assert tok_tf.mask_token == "<mask>"
        assert tok_tf.mask_token_id == 4
        return tok_tf

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
        tok_tf.backend_tokenizer.pre_tokenizer = Split(Regex(regex), "isolated")
        return tok_tf

    else:
        # Fall back to a HuggingFace Tokenizer
        from transformers import AutoTokenizer

        from smirk import SmirkTokenizerFast

        AutoTokenizer.register(
            "SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast
        )

        tok_tf = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=".cache",  # Cache Tokenizer in working directory
            **kwargs,
        )
        ensure_special_tokens(tok_tf)
        return tok_tf


def ensure_special_tokens(tok: PreTrainedTokenizerBase):
    tok.add_special_tokens(
        {
            "unk_token": tok.unk_token or match_special_tokens(tok, "[UNK]", "<unk>"),
            "mask_token": tok.mask_token
            or match_special_tokens(tok, "[MASK]", "<mask>"),
            "pad_token": tok.pad_token or match_special_tokens(tok, "[PAD]", "<pad>"),
        }
    )


def match_special_tokens(tok: PreTrainedTokenizerBase, *candidates: list[str]):
    vocab = tok.get_vocab()
    for c in candidates:
        if c in vocab.keys():
            return c
    return candidates[0]


def regex_smiles_tokenizer(vocab: dict, regex: str, unk_token: str = "[UNK]"):
    from tokenizers import Regex, Tokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import Strip
    from tokenizers.pre_tokenizers import Split

    tok = Tokenizer(WordLevel(vocab, unk_token))
    tok.normalizer = Strip()
    tok.pre_tokenizer = Split(Regex(regex), "isolated")
    tok.decoder = ByteLevel()  # When decoding don't add spaces between tokens
    tok_tf = PreTrainedTokenizerFast(tokenizer_object=tok, add_special_tokens=True)
    ensure_special_tokens(tok_tf)
    return tok_tf


def download(url: str, path: Path):
    import urllib.request

    with urllib.request.urlopen(url) as fid:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as out:
            out.write(fid.read())


def find_member(tar, name):
    for m in tar.getmembers():
        # Strip the first path component
        p = Path(*str(m.name).split("/")[1:])
        if str(p) == name:
            return m

    return None


def extract_file(archive, name, output: Path, **kwargs):
    import tarfile
    from shutil import move
    from tempfile import TemporaryDirectory

    with tarfile.open(archive) as fid:
        member = find_member(fid, name)
        assert member is not None
        with TemporaryDirectory() as tmp:
            fid.extract(member, filter="data", path=tmp, **kwargs)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            move(str(Path(tmp).joinpath(member.name)), str(output))


def vocab_from_words(file: str, add_unk_token=True, unk_token: str = "[UNK]"):
    import json

    if str(file).endswith(".json"):
        with open(file, "r") as fid:
            vocab = json.load(fid)

    elif str(file).endswith(".txt"):
        with open(file, "r") as fid:
            vocab = {}
            for id, token in enumerate(fid.readlines()):
                vocab[token.strip()] = id
    else:
        raise RuntimeError("Unknown extension", file)

    if add_unk_token:
        vocab[unk_token] = len(vocab)
    return vocab


def cached_github_archive(repo, commit, file):
    cache = Path(__file__).parent.parent.parent.joinpath(".cache", "smiles-tokenizers")
    cache.mkdir(exist_ok=True, parents=True)

    repo_cache = cache.joinpath(repo.replace("/", "-"), commit)
    archive = repo_cache.joinpath("archive.tar.gz")
    if not archive.is_file():
        url = f"https://github.com/{repo}/archive/{commit}.tar.gz"
        download(url, archive)

    cached_path = repo_cache.joinpath(file)
    if not cached_path.is_file():
        extract_file(archive, file, cached_path)

    return cached_path


def rdkit_canonical(smi: str) -> str:
    """Canonicalize a SMILES encoding using rdkit"""
    from rdkit import Chem

    try:
        return Chem.CanonSmiles(smi)
    except Exception:
        return None
