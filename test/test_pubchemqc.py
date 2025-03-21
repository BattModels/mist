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
        "opt", "pubchem-qc", "pubchemqc_jcim2017"
    )
    if dir.exists():
        return dir
    return None


@pytest.mark.skipif(
    pubchem_qc_dataset_path() is None, reason="Missing PubChem QC Dataset"
)
@pytest.fixture(scope="session")
def pubchem_qc_examples():
    dir = pubchem_qc_dataset_path()
    assert dir is not None
    ds = load_dataset(
        "arrow",
        data_files=[str(dir.joinpath("*.arrow"))],
        keep_in_memory=False,
        split="train",
    )
    return ds.take(10)


@pytest.fixture(params=list(range(10)))
def example_mol(request, pubchem_qc_examples):
    row = pubchem_qc_examples[request.param]
    mol = construct_mol(
        atomic_numbers=row["atomic-numbers"],
        positions=row["atomic-coordinates"],
        bonds=row["bond-connections"],
        bond_orders=row["bond-order"],
    )
    yield mol


def test_mol(example_mol):
    assert isinstance(example_mol, Chem.Mol)
    assert example_mol.GetConformer() is not None
    for atom in example_mol.GetAtoms():
        assert atom.GetNoImplicit()
        assert atom.GetNumImplicitHs() == 0


def test_construct_mol(pubchem_qc_examples):
    data = pubchem_qc_examples[0]
    mol = construct_mol(
        atomic_numbers=data["atomic-numbers"],
        positions=data["atomic-coordinates"],
        bonds=data["bond-connections"],
        bond_orders=data["bond-order"],
    )
    assert isinstance(mol, Chem.Mol)
    assert 3 * mol.GetNumAtoms() == len(data["atomic-coordinates"])
    for idx, order in enumerate(data["bond-order"]):
        sdx, edx = data["bond-connections"][2 * idx : 2 * idx + 2]
        bond = mol.GetBondBetweenAtoms(sdx, edx)
        assert bond is not None
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            assert order in [1, 2]  # Allow for kekule
        else:
            assert bond.GetBondTypeAsDouble() == order

    conf = mol.GetConformer()
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        assert pos.x == data["atomic-coordinates"][3 * idx]
        assert pos.y == data["atomic-coordinates"][3 * idx + 1]
        assert pos.z == data["atomic-coordinates"][3 * idx + 2]

    # Add properties
    charge = [random.normalvariate(0, 1) for _ in range(mol.GetNumAtoms())]
    mol = add_atomic_properties(mol, {"charge": charge})
    for idx, q in enumerate(charge):
        assert mol.GetAtomWithIdx(idx).GetDoubleProp("charge") == q


def test_aligned_tokenized(pubchem_qc_examples):
    data = pubchem_qc_examples[0]
    mol = construct_mol(
        atomic_numbers=data["atomic-numbers"],
        positions=data["atomic-coordinates"],
        bonds=data["bond-connections"],
        bond_orders=data["bond-order"],
    )

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
    net_charge = torch.masked.masked_tensor(
        out["token_target"], out["token_target_mask"]
    ).sum()
    assert ref_net_charge.isclose(net_charge, atol=1e-6).item()


@pytest.fixture(
    params=[
        "c1ccc(c(c1)C[P](=O)O)O",
        "O=C1c2ccccc2C(=O)N1C3CCC(=O)NC3=O",
        "CN3[C@H]1CC[C@@H]3C[C@@H](C1)OC(=O)C(CO)c2ccccc2",
        "O=C(OC)[C@H]2[C@@]3(CC[C@H]4C(=O)O[C@H](c1ccoc1)C[C@@]4([C@H]3C(=O)[C@@H](OC(=O)C)C2)C)C",
    ]
)
def smear_hydrogen_fixture(request):
    smi = request.param
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("atom_index", idx)

    # Add Hs and add indices
    mol_hs = Chem.AddHs(mol)
    idx = mol.GetNumAtoms()
    assert mol_hs.GetNumAtoms() > mol.GetNumAtoms()
    for atom in mol_hs.GetAtoms():
        atom.SetNoImplicit(True)
        if not atom.HasProp("atom_index"):
            atom.SetIntProp("atom_index", idx)
            idx += 1

    assert mol_hs.GetNumAtoms() == idx
    n_atoms = mol_hs.GetNumAtoms()
    target = {
        "v1": [random.random() for _ in range(n_atoms)],
        "v2": [random.random() for _ in range(n_atoms)],
    }
    mol_hs_prop = add_atomic_properties(mol_hs, target)
    assert mol_hs_prop.GetNumAtoms() == mol_hs.GetNumAtoms()

    return mol, mol_hs_prop, target


def test_smear_hydrogens(smear_hydrogen_fixture):
    _, mol_hs_prop, targets = smear_hydrogen_fixture
    mol_smear = pubchem_qc.smear_hydrogen_targets(mol_hs_prop, targets.keys())
    for atom in mol_smear.GetAtoms():
        atom_idx = atom.GetIntProp("atom_index")
        for target, target_values in targets.items():
            assert atom.HasProp(target)
            assert atom.GetDoubleProp(target) == target_values[atom_idx]
            assert atom.HasProp(f"hs_{target}") == atom.GetBoolProp("has_hs_target")


def test_collate_partial_charges(pubchem_qc_examples, caplog):
    caplog.set_level(logging.DEBUG)
    for row in pubchem_qc_examples:
        out = collate_partial_charges(row, tokenizer=SmirkTokenizerFast())
        for f in [
            "input_ids",
            "target",
            "target_mask",
            "token_target",
            "token_target_mask",
        ]:
            assert f in out
        logging.info({"smi": row["smiles"], **out})


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
