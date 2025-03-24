import logging
import pytest
import torch
import json
import random
from statistics import mean
from pathlib import Path
from rdkit import Chem
from datasets import load_dataset
from electrolyte_fm.data_modules import pubchem_qc
from electrolyte_fm.data_modules.pubchem_qc import (
    PubChemQC,
    annotated_tokens,
    construct_mol,
    add_atomic_properties,
    collate_partial_charges,
    SmiTokenType,
)
from smirk import SmirkTokenizerFast
from .test_dataset import check_datamodule


def pubchem_qc_dataset_path():
    dir = Path(__file__).parent.parent.joinpath(
        "opt", "pubchem-qc", "pubchemqc_jcim2017-split", "train"
    )
    if dir.exists():
        return dir
    return None


NUM_PUBCHEM_QC_EXAMPLES = 32


@pytest.fixture(scope="session")
def __pubchem_qc_examples():
    dir = pubchem_qc_dataset_path()
    if dir is None:
        pytest.skip("Missing PubChem QC Dataset")
    ds = load_dataset(
        "arrow",
        data_files=[str(dir.joinpath("*.arrow"))],
        keep_in_memory=False,
        split="train",
    )
    return ds.take(NUM_PUBCHEM_QC_EXAMPLES)


@pytest.fixture(params=list(range(NUM_PUBCHEM_QC_EXAMPLES)))
def pubchem_qc_example(request, __pubchem_qc_examples):
    return __pubchem_qc_examples[request.param]


@pytest.fixture()
def example_mol(pubchem_qc_example):
    row = pubchem_qc_example
    mol = construct_mol(
        atomic_numbers=row["atomic-numbers"],
        positions=row["atomic-coordinates"],
        bonds=row["bond-connections"],
        bond_orders=row["bond-order"],
    )
    yield {"mol": mol, **row}


def test_mol(example_mol):
    mol = example_mol["mol"]
    assert isinstance(mol, Chem.Mol)
    assert mol.GetConformer() is not None
    for atom, an in zip(mol.GetAtoms(), example_mol["atomic-numbers"]):
        assert atom.GetNoImplicit()
        assert atom.GetNumImplicitHs() == 0
        assert atom.GetAtomicNum() == an


def test_construct_mol(example_mol):
    mol = example_mol["mol"]
    assert isinstance(mol, Chem.Mol)
    assert 3 * mol.GetNumAtoms() == len(example_mol["atomic-coordinates"])
    for idx, order in enumerate(example_mol["bond-order"]):
        sdx, edx = example_mol["bond-connections"][2 * idx : 2 * idx + 2]
        bond = mol.GetBondBetweenAtoms(sdx, edx)
        assert mol.GetAtomWithIdx(sdx).GetIntProp("atom_index") == sdx
        assert mol.GetAtomWithIdx(edx).GetIntProp("atom_index") == edx
        assert bond is not None
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            assert order in [1, 2]  # Allow for kekule
        else:
            assert bond.GetBondTypeAsDouble() == order

    conf = mol.GetConformer()
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        assert pos.x == example_mol["atomic-coordinates"][3 * idx]
        assert pos.y == example_mol["atomic-coordinates"][3 * idx + 1]
        assert pos.z == example_mol["atomic-coordinates"][3 * idx + 2]

    # Add properties
    charge = [random.normalvariate(0, 1) for _ in range(mol.GetNumAtoms())]
    mol = add_atomic_properties(mol, {"charge": charge})
    for idx, q in enumerate(charge):
        assert mol.GetAtomWithIdx(idx).GetDoubleProp("charge") == q


def test_aligned_tokenized(example_mol):
    mol = example_mol["mol"]

    # Construct fake charge data
    charge = [random.normalvariate(0, 1) for _ in range(mol.GetNumAtoms())]
    avg_charge = mean(charge)
    charge = [q - avg_charge for q in charge]
    mol = add_atomic_properties(mol, {"charge": charge})

    # Annotate tokens with data
    tokenizer = SmirkTokenizerFast()
    out = annotated_tokens(mol, targets=["charge"], tokenizer=tokenizer)
    assert isinstance(out["token_target"], torch.Tensor)
    assert out["token_target"].shape == (len(out["input_ids"]), 2)
    assert out["token_target_mask"].shape == (len(out["input_ids"]), 2)
    ref_net_charge = torch.tensor(charge).sum()
    net_charge = (
        torch.masked.masked_tensor(out["token_target"], out["token_target_mask"])
        .sum()
        .get_data()
    )
    assert ref_net_charge.isclose(net_charge, atol=1e-5)


@pytest.fixture()
def example_mol_target(example_mol):
    target = {
        "mulliken": example_mol["partial-charge-mulliken"],
        "lowdin": example_mol["partial-charge-lowdin"],
    }
    mol = example_mol["mol"]
    mol_with_target = add_atomic_properties(mol, target)
    assert mol_with_target.GetNumAtoms() == mol.GetNumAtoms()
    assert all(
        a.GetSmarts() == b.GetSmarts()
        for a, b in zip(mol.GetAtoms(), mol_with_target.GetAtoms())
    )
    assert "mol_with_target" not in example_mol
    assert "target" not in example_mol
    return {"mol_with_target": mol_with_target, "target": target, **example_mol}


def test_annotate_position(example_mol_target):
    mol_with_target = example_mol_target["mol_with_target"]
    tokenizer = SmirkTokenizerFast()
    out = annotated_tokens(
        mol_with_target,
        targets=example_mol_target["target"],
        include_3d=True,
        tokenizer=tokenizer,
    )
    token_target = out["token_target"]
    token_mask = out["token_target_mask"]
    assert token_target.shape == token_mask.shape
    assert token_target.shape[-1] == 2 * len(example_mol_target["target"]) + 3
    pos_mask = token_mask[:, -3:].all(-1)
    pos = token_target[pos_mask, -3:]
    logging.debug(
        {
            "smi": out["smi"],
            "token_target": token_target,
            "token_mask": token_mask,
            "masked_pos": pos,
        }
    )
    assert pos_mask.any()


def test_smear_hydrogens(example_mol_target):
    mol_with_target = example_mol_target["mol_with_target"]
    targets = example_mol_target["target"]
    mol_smear = pubchem_qc.smear_hydrogen_targets(mol_with_target, targets.keys())
    h_count = sum(an == 1 for an in example_mol_target["atomic-numbers"])
    h_atoms = sum(atom.GetAtomicNum() == 1 for atom in mol_smear.GetAtoms())
    assert h_atoms <= h_count
    mol_has_hs_target = False
    for atom in mol_smear.GetAtoms():
        atom_idx = atom.GetIntProp("atom_index")
        for target, target_values in targets.items():
            assert atom.HasProp(target)
            assert atom.GetDoubleProp(target) == target_values[atom_idx]
            assert atom.HasProp(f"hs_{target}") == atom.GetBoolProp("has_hs_target")
            mol_has_hs_target |= atom.GetBoolProp("has_hs_target")

    # Either no hydrogen atoms were removed or the mol has a hs_target
    assert (h_atoms == h_count) != mol_has_hs_target
    logging.debug(
        {"smi": Chem.MolToSmiles(mol_smear), "mol_has_hs_target": mol_has_hs_target}
    )


def test_smear_smi(example_mol_target):
    # h_atoms should match the h_count from the smiles
    mol_with_target = example_mol_target["mol_with_target"]
    mol_smear = pubchem_qc.smear_hydrogen_targets(
        mol_with_target, example_mol_target["target"].keys()
    )
    h_atoms = sum(atom.GetAtomicNum() == 1 for atom in mol_smear.GetAtoms())
    mol_smi = Chem.MolFromSmiles(Chem.MolToSmiles(mol_with_target))
    if mol_smi is None:
        # Not ideal, but with only one molecule triggering its fine
        pytest.xfail("molecule failes to convert to smiles")
    h_atoms_smi = sum(atom.GetAtomicNum() == 1 for atom in mol_smi.GetAtoms())
    assert h_atoms_smi == h_atoms


def test_collate_partial_charges(pubchem_qc_example):
    out = collate_partial_charges(pubchem_qc_example, tokenizer=SmirkTokenizerFast())
    for f in [
        "input_ids",
        "target",
        "target_mask",
        "token_target",
        "token_target_mask",
    ]:
        assert f in out

    # Check has_hs_target was populated
    h_atom = sum(an == 1 for an in pubchem_qc_example["atomic-numbers"])
    logging.debug(out["smi"])
    h_smi = sum(
        atom.GetAtomicNum() == 1
        for atom in Chem.MolFromSmiles(out["smi"], sanitize=False).GetAtoms()
    )
    token_target_mask = out["token_target_mask"]
    assert token_target_mask.shape[1] == 4
    assert token_target_mask.shape[0] == len(out["input_ids"])
    logging.debug(
        {
            "smi": out["smi"],
            "token_target_mask": token_target_mask,
            "h_atom": h_atom,
            "h_smi": h_smi,
        }
    )
    if h_smi < h_atom:
        # Both central and hs_targets should be defined for at least one atom
        assert token_target_mask.any(0).all()
    else:
        # No hs_targets should be defined for all atoms
        assert not token_target_mask[:, [1, 3]].any()
        assert token_target_mask[:, [0, 2]].any(0).all()


def test_smiles_numbering():
    # Sanity checking understanding of _smilesAtomOutputOrder
    mol = Chem.MolFromSmiles("OCC")
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("atom_index", idx)
    smi = Chem.MolToSmiles(mol)
    assert smi == "CCO"
    smi_order = [int(c) for c in mol.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")]
    assert all(a == b for a, b in zip(smi_order, [2, 1, 0]))

    # Check atoms don't get reordered
    for idx, atom in enumerate(mol.GetAtoms()):
        assert atom.GetIntProp("atom_index") == idx


@pytest.mark.skipif(
    pubchem_qc_dataset_path() is None, reason="Missing PubChem QC Dataset"
)
def test_pubchem_qc_dm():
    dm = PubChemQC(str(pubchem_qc_dataset_path()), num_workers=8)
    check_datamodule(dm, limit_batches=10)


@pytest.mark.parametrize(
    "smi,token_types",
    [
        ("CCC", [SmiTokenType.Element, SmiTokenType.Element, SmiTokenType.Element]),
        (
            "CC(C)C",
            [
                SmiTokenType.Element,
                SmiTokenType.Element,
                SmiTokenType.Structure,
                SmiTokenType.Element,
                SmiTokenType.Structure,
                SmiTokenType.Element,
            ],
        ),
        (
            "C[C@@H]",
            [
                SmiTokenType.Element,  # C
                SmiTokenType.Bracket,  # [
                SmiTokenType.Element,  # C
                SmiTokenType.Unknown,  # @@
                SmiTokenType.ExplicitHydrogen,  # H
                SmiTokenType.Bracket,  # ]
            ],
        ),
        (
            "[HH]",
            [
                SmiTokenType.Bracket,
                SmiTokenType.Element,
                SmiTokenType.ExplicitHydrogen,
                SmiTokenType.Bracket,
            ],
        ),
        (
            "[Co@OH9H3]",
            [
                SmiTokenType.Bracket,
                SmiTokenType.Element,
                SmiTokenType.Unknown,
                SmiTokenType.Unknown,
                SmiTokenType.ExplicitHydrogen,
                SmiTokenType.Unknown,
                SmiTokenType.Bracket,
            ],
        ),
    ],
)
def test_smi_token_type(smi: str, token_types: list[SmiTokenType]):
    tok = SmirkTokenizerFast()
    tokens = tok(smi)["input_ids"]
    assert isinstance(tokens, list)
    assert isinstance(tokens[0], int)
    assert len(list(pubchem_qc.smi_token_type(tok, tokens))) == len(tokens)
    for idx, (token_type, ref) in enumerate(
        zip(pubchem_qc.smi_token_type(tok, tokens), token_types)
    ):
        token = tok.convert_ids_to_tokens(tokens[idx])
        assert token_type == ref, (
            f"Wrong label for {token} at pos {idx}: {str(token_type)} != {str(ref)}"
        )
