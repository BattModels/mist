from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import smirk
from electrolyte_fm.tokenize.spe import PreTrainedSPETokenizer, pretrained_spe_tokenizer
from electrolyte_fm.utils.tokenizer import load_tokenizer, rdkit_canonical

SMILE_TOKENIZER = [
    "smirk",
    "ibm/MoLFormer-XL-both-10pct",
]


@pytest.fixture(scope="module", params=SMILE_TOKENIZER)
def smile_tokenizer(request):
    return load_tokenizer(request.param)


STANDARD_SMILES = [
    "CC[N+](C)(C)Cc1ccccc1Br",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN(CCC1=CNC2=C1C=CC=C2)C",
    "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
]

# Tokenizers not actively used for training
OTHER_SMILES_TOKENIZERS = [
    "SmilesPE/SPE_ChEMBL",
    "devalab/molgpt-moses",
    "devalab/molgpt-guacamol",
    "MolecularAI/Chemformer",
    "MolecularAI/Chemformer-downstream",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
    "sagawa/ReactionT5-yield-prediction",
    "rxn4chemistry/rxn_yields",
    "rxn4chemistry/rxnfp",
    "ChangwenXu98/TransPolymer",
]


@pytest.fixture(scope="module", params=OTHER_SMILES_TOKENIZERS)
def other_smiles_tokenizer(request):
    return load_tokenizer(request.param)


SELFIES_TOKENIZERS = [
    "smirk-selfies",
    "HUBioDataLab/SELFormer",
]


@pytest.fixture(scope="module", params=SELFIES_TOKENIZERS)
def selfies_tokenizers(request):
    return load_tokenizer(request.param)


def test_well_behaved_tokenizer(other_smiles_tokenizer):
    code = other_smiles_tokenizer("CCO")
    assert other_smiles_tokenizer.unk_token_id is not None
    assert other_smiles_tokenizer.unk_token_id not in code["input_ids"]
    check_encoding(other_smiles_tokenizer, ["CCO", "C-C-O", "CC(C)C(=O)C(C)C"])


def test_well_behaved_selfies(selfies_tokenizers):
    assert selfies_tokenizers.unk_token_id is not None
    for selfie in ["[C][C][O]", "[O][=C][C][=C][C][=C][C][=C][Ring1][=Branch1]"]:
        code = selfies_tokenizers(selfie)
        assert selfies_tokenizers.unk_token_id not in code["input_ids"]


""" The following SMILES strings are permissible per the OpenSMILES spec but known to not be parsed by rdkit """
RDKIT_EXPECTED_FAILURES = [
    "[02H]",
    "[002H]",
    "[NH4+:005]",
    "C1CCCCC%01",
    "C%00CCCCC%00",
    "Oc1cc(.NCCO)ccc1",
    "C%01CCCCC%01",
]


@pytest.mark.parametrize("file", ["opensmiles.smi", "smiles.txt"])
def test_canonical_smiles(file):
    smirk_test = Path(__file__).parent.parent.joinpath("smirk", "test")
    with open(smirk_test.joinpath(file), "r") as fid:
        for smi in fid.readlines():
            smi = smi.strip()
            if smi.startswith("#"):
                continue
            canon = rdkit_canonical(smi)
            if smi not in RDKIT_EXPECTED_FAILURES:
                assert canon is not None, f"failed to canonicalize {smi}"
            else:
                assert canon is None, f"expected failure for {smi}. got {canon}"


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
    if isinstance(smile_tokenizer, PreTrainedSPETokenizer):
        pytest.xfail("Out of vocab for SPETokenizer")
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


def test_spe_setup():
    tokenizer = pretrained_spe_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    vocab = tokenizer.get_vocab()
    assert "xxfake" not in vocab
    assert "[BOS]" in vocab
    assert "[N+]" in vocab
