import json
import urllib
from tempfile import TemporaryDirectory
from itertools import chain

import pytest
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import smirk
from electrolyte_fm.utils.tokenizer import load_tokenizer

SMILE_TOKENIZER = [
    "smirk",
    "ibm/MoLFormer-XL-both-10pct",
]

STANDARD_SMILES = [
    "CC[N+](C)(C)Cc1ccccc1Br",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN(CCC1=CNC2=C1C=CC=C2)C",
    "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
]

# Tokenizers not actively used for training
OTHER_SMILES_TOKENIZERS = [
    "character",
    "SmilesPE/SPE_ChEMBL",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "devalab/molgpt-moses",
    "MolecularAI/Chemformer",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
    "bm2-lab/X-MOL",
]


def flaky_load_tokenizer(name_or_path, *args, **kwargs):
    try:
        return load_tokenizer(name_or_path, *args, **kwargs)

    except (json.decoder.JSONDecodeError, urllib.error.HTTPError):
        if name_or_path in ["SmilesPE/SPE_ChEMBL"]:
            pytest.xfail("SPE tokenizer not available (flaky download)")
        raise

    except ImportError:
        pytest.xfail(f"Dependency not installed for {name_or_path}")


@pytest.fixture(scope="module", params=SMILE_TOKENIZER)
def smile_tokenizer(request):
    return flaky_load_tokenizer(request.param)


@pytest.mark.parametrize("name", chain(SMILE_TOKENIZER, OTHER_SMILES_TOKENIZERS))
def test_well_behaved_tokenizer(name):
    tokenizer = flaky_load_tokenizer(name)
    code = tokenizer("CCO")
    tokens = tokenizer.tokenize("CCO")
    assert isinstance(tokens, list) and len(tokens) >= 1 and isinstance(tokens[0], str)
    assert tokenizer.mask_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id not in code["input_ids"]

    for special in [tokenizer.mask_token, tokenizer.pad_token, tokenizer.unk_token]:
        assert special in tokenizer.all_special_tokens
        assert tokenizer.encode(special)[0] in tokenizer.all_special_ids

    vocab = tokenizer.get_vocab()
    assert vocab[tokenizer.mask_token] == tokenizer.mask_token_id

    check_encoding(tokenizer, ["CCO", "C-C-O", "CC(C)C(=O)C(C)C"])


@pytest.mark.parametrize("name", chain(SMILE_TOKENIZER, OTHER_SMILES_TOKENIZERS))
def test_oov_tokens(name):
    """Check that the unknown token is emitted for OOV tokens"""
    tok = flaky_load_tokenizer(name)

    def check_oov(tokenizer, smi):
        code = tokenizer(smi)["input_ids"]
        assert tok.unk_token_id in code
        assert tok.decode(code, skip_special_tokens=False) != smi
        assert str(tok.unk_token) in tok.decode(code)

    # Some Tokenizers don't emit the unknown token no matter what the input is.
    # Check that the tokenizer fails the test, then mark it with xfail
    if name == "ibm/MoLFormer-XL-both-10pct":
        assert tok.unk_token_id not in tok("😬")["input_ids"]
        pytest.xfail("MoLFormer strips unknown tokens pre-tokenizer")
    elif name in ["seyonec/ChemBERTa-zinc-base-v1", "ChangwenXu98/TransPolymer"]:
        assert tok.unk_token_id not in tok("⛰️ ⋙ 🏖️")["input_ids"]
        pytest.xfail("open vocab model")
    elif name == "character":
        check_oov(tok, "⚛️")
        assert tok.unk_token_id not in tok("ZZ[Zz]")["input_ids"]
        pytest.xfail("character-level tokenizer")

    check_oov(tok, "Zz")
    check_oov(tok, "[Zz]")
    check_oov(tok, "[Zz&3]")


def test_vocab_size(smile_tokenizer):
    # vocab size (Size of model vocab without added tokens)
    # should be smaller (or equal if the model knows of all added tokens)
    # than the length of of the tokenizer
    assert smile_tokenizer.vocab_size <= len(smile_tokenizer)


def test_pretrained_smirk():
    tok = smirk.SmirkTokenizerFast()

    with TemporaryDirectory() as dir:
        tok.save_pretrained(dir)
        loaded = load_tokenizer(str(dir))
        assert isinstance(loaded, smirk.SmirkTokenizerFast)
        assert tok.to_str() == loaded.to_str()


def check_encoding(tokenizer: PreTrainedTokenizerBase, batch: list[str]):
    # Check output
    code = tokenizer(batch)
    assert isinstance(code, BatchEncoding)
    assert "input_ids" in code.keys()
    assert isinstance(code["input_ids"], list)
    assert isinstance(code["input_ids"][0], list)
    assert isinstance(code["input_ids"][0][0], int)

    # Check reverse
    assert len(code["input_ids"]) == len(batch)
    out = tokenizer.batch_decode(code["input_ids"], skip_special_tokens=True)
    assert len(out) == len(batch)
    assert out == batch


def test_standard_smiles(smile_tokenizer):
    check_encoding(smile_tokenizer, STANDARD_SMILES)


def test_compounds_smiles(smile_tokenizer):
    check_encoding(
        smile_tokenizer,
        [
            "[Sc+3].[OH-].[OH-].[OH-]",
            "[Li+].F[P-](F)(F)(F)(F)F",
        ],
    )


def test_mlm_tokenizer(smile_tokenizer):
    collate = DataCollatorForLanguageModeling(smile_tokenizer, mlm_probability=0.5)
    code = [smile_tokenizer(smile) for smile in STANDARD_SMILES]
    print(code)
    collated_batch = collate(code)
    print(collated_batch)

    # Should pad to longest
    max_length = max((len(x["input_ids"]) for x in code))
    for k in [
        "input_ids",
        # "token_type_ids",
        "attention_mask",
        "labels",
    ]:
        assert k in collated_batch.keys()
        assert collated_batch[k].size() == (len(STANDARD_SMILES), max_length)


@pytest.mark.xfail(
    strict=False, raises=urllib.error.HTTPError, reason="Flaky downloads"
)
def test_spe_setup():
    try:
        from electrolyte_fm.tokenize.spe import pretrained_spe_tokenizer
    except ImportError:
        pytest.skip("SmilesPE not installed")

    tokenizer = pretrained_spe_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    vocab = tokenizer.get_vocab()
    assert "xxfake" not in vocab
    assert "[BOS]" in vocab
    assert "[N+]" in vocab
    assert "[N+]" in vocab


def test_xmol_tokenizer():
    # Reference tokenizer from X-MOL
    # https://github.com/bm2-lab/X-MOL/blob/a64cd4222ab819326767224d91fa8605f52f4fc4/FT_to_prediction/tokenization.py#L152-L176
    def tokenize(smi):
        tokens = []
        dc_a = ("l", "r")
        marker = 0  # 1,2 indicates that the last token is not complete
        for c in smi:
            if marker == 0:
                if c in dc_a:
                    tokens[-1] += c
                else:
                    tokens.append(c)
                    if c == "[":
                        marker = 1
                    elif c == "%":
                        marker = 2
                        marker_l = 2  # indicates that remain length of %**
            else:
                tokens[-1] += c
                if marker == 1:
                    if c == "]":
                        marker = 0
                elif marker == 2:
                    marker_l -= 1
                    if marker_l == 0:
                        marker = 0
        return tokens

    tok = load_tokenizer("bm2-lab/X-MOL")
    corpus = [
        *STANDARD_SMILES,
        # The following additional molecules were taken from
        # the X-MOL paper to confirm our regex replication matches
        # the original state-machine implementation
        # doi:10.1016/j.scib.2022.01.029
        "CCn1cc(NC(=O)C2=Cc3ccccc3OC=C2)ccc1=O",
        "O=S(=O)(N[C@H]1C[C@H]2CC[C@@H]1C2)c1cccc2cccnc12",
        "CS[C@H]1CC[C@@H](NC(=O)c2cnn3c(C)c(C)sc23)C1",
        "C[C@@H]1Cc2ccccc2N1CC(=O)Nc1ccc(F)c(C#N)c1",
        "CCS(=O)(=O)N1CC=C(c2c(C)[nH]c3ccccc23)CC1",
        "O=C(O)c1cccc(P(=O)(O)O)c1",
        "c1cccc(c1C(O)=O)C(=O)C",
        # Base sanity
        "",
        "C",
        "Cl",
        "Br",
        # multiple halogens / dc_a gluing
        "CClBr",
        "Clrl",
        "Cll",
        "Crl",
        "Crlr",
        "ClBr",
        "BrCl",
        "Cl(Cl)Cl",
        # generic structure sanity
        "C.C",
        "C=C",
        "C#N",
        "C(C)(C)C",
        "c1ccccc1Cl",
        # Bracket atoms
        "[NH3+]",
        "[13CH2-]",
        "[NH3+]l",
        "[NH3+]ll",
        "[13CH2-]r",
        "[13CH2-]rl",
        # Bracket weirdness / unterminated
        "C[",
        "[C",
        "C[NH3+",
        "[C[NH3+",
        "[C%",
        "[C%1",
        "[C%12",
        # nested-ish bracket weirdness
        "[C[NH3+]]",
        "[C[NH3+]]l",
        "[C[NH3+]]ll",
        "[[C]]",
        "[[C]",
        "[]",
        "[]C",
        # Stray closing bracket
        "C]",
        "]C",
        "]",
        # Ring closure cases
        "C1CCCCC1",  # plain ring digits
        "c1ccccc1Cl",
        # Percent ring opening (marker=2 logic)
        "%",  # literally just a percent
        "%1",
        "%12",
        "C%12",
        "C%12C",
        "C%9",
        "C%9l",
        "C%12CC%12",
        # Incomplete % sequences
        "C%",
        "C%1",
        # Percent + dc_a interactions
        "%12l",
        "%12ll",
        "C%1l",
        "C%1ll",
        "C%l",
        "C%lr",
    ]
    for smi in corpus:
        ref_tokens = tokenize(smi)
        tokens = tok.tokenize(smi)
        if "[UNK]" in tokens:
            # Just split to preserve the UNKs
            tokens = tok._tokenizer.pre_tokenizer.pre_tokenize_str(smi)
            tokens = [tok for tok, offset in tokens]
        assert tokens == ref_tokens
