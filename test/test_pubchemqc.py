import json
import itertools
import logging
import random
from pathlib import Path
from statistics import mean
from typing import Mapping, Any
from math import cos, sin

import pytest
import torch
from datasets import load_dataset
from rdkit import Chem
from smirk import SmirkTokenizerFast

from electrolyte_fm.data_modules import pubchem_qc
from electrolyte_fm.models.normalize import AbstractNormalizer, Standardize
from electrolyte_fm.data_modules.pubchem_qc import (
    PubChemQC,
    SmiTokenType,
    add_atomic_properties,
    annotated_tokens,
    collate_partial_charges,
    construct_mol,
    mol_from_prediction,
)
from electrolyte_fm.models.token_level import distance_matrix_loss


def pubchem_qc_dataset_path():
    dir = Path(__file__).parent.parent.joinpath(
        "opt", "pubchem-qc", "pubchemqc_jcim2017-split"
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
        data_files=[str(dir.joinpath("train", "*.arrow"))],
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

    assert "token_coords" in out
    assert "token_coords_mask" in out


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
        tokenizer=tokenizer,
    )
    token_target = out["token_target"]
    token_mask = out["token_target_mask"]
    assert token_target.shape == token_mask.shape
    assert token_target.shape[-1] == 2 * len(example_mol_target["target"])

    token_coords = out["token_coords"]
    token_coords_mask = out["token_coords_mask"]
    assert token_coords_mask.any()
    assert token_coords_mask.ndim == 1
    assert token_coords_mask.shape[0] == token_coords.shape[0]


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


@pytest.mark.skipif(
    pubchem_qc_dataset_path() is None, reason="Missing PubChem QC Dataset"
)
@pytest.mark.parametrize(
    "include_3d,randomize,include_topo_dist",
    [
        (True, True, False),
        (False, False, False),
        ("as-target", True, False),
        ("as-target", True, True),
    ],
)
def test_datamodule(include_3d, randomize, include_topo_dist):
    dir = pubchem_qc_dataset_path()
    assert dir is not None
    dm = PubChemQC(
        str(dir),
        include_3d=include_3d,
        randomize=randomize,
        include_topo_dist=include_topo_dist,
        batch_size=16,
    )
    dm.prepare_data()
    dm.setup("fit")

    def check_dataloader(dl):
        for batch in itertools.islice(dl, 16):
            assert isinstance(batch, Mapping)
            for col in [
                "input_ids",
                "attention_mask",
                "target",
                "target_mask",
                "token_target",
                "token_target_mask",
            ]:
                assert col in batch
                assert isinstance(batch[col], torch.Tensor)

            B, S = batch["input_ids"].shape
            for target in ["target", "token_target"]:
                assert batch[target].shape == batch[target + "_mask"].shape

            if include_3d is True:
                assert batch["token_target"].shape == (B, S, 7)
                assert batch["token_target_mask"].shape == (B, S, 7)
                assert batch["token_target_mask"].any(1)[:, -3:].all()
            elif include_3d == "as-target":
                assert batch["token_coords"].shape == (B, S, 3)
                assert batch["token_coords_mask"].shape == (B, S)

            else:
                assert batch["token_target_mask"].shape[-1] == 4

    check_dataloader(dm.train_dataloader())
    check_dataloader(dm.val_dataloader())
    check_dataloader(dm.test_dataloader())


@pytest.mark.skipif(
    pubchem_qc_dataset_path() is None, reason="Missing PubChem QC Dataset"
)
def test_normalize():
    dir = pubchem_qc_dataset_path()
    assert dir is not None
    dm = PubChemQC(str(dir), include_3d=True, batch_size=16)
    dm.prepare_data()
    dm.setup("fit")

    transform = AbstractNormalizer.get("standardize", num_outputs=9)
    assert isinstance(transform, Standardize)
    ds = dm.train_dataset.take(100)
    state = transform.fit(ds.select_columns(["target", "target_mask"]))
    assert state["mean"].shape == (len(dm.seq_targets),)
    assert state["std"].shape == (len(dm.seq_targets),)

    # Repeat for token targets
    transform = AbstractNormalizer.get("standardize", num_outputs=7)
    assert isinstance(transform, Standardize)
    ds_token = ds.select_columns(["token_target", "token_target_mask"])
    ds_token = ds_token.rename_columns(
        {"token_target": "target", "token_target_mask": "target_mask"}
    )
    state = transform.fit(ds_token)
    assert state["mean"].shape == (2 * len(dm.token_targets) + 3,)
    assert state["std"].shape == (2 * len(dm.token_targets) + 3,)


def test_mol_from_prediction(pubchem_qc_example: dict[str, Any]):
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    include_3d = True
    out = collate_partial_charges(
        pubchem_qc_example,
        tokenizer,
        include_3d=include_3d,
        randomize=False,
    )

    # Reference
    mol_ref = construct_mol(
        atomic_numbers=pubchem_qc_example["atomic-numbers"],
        positions=pubchem_qc_example["atomic-coordinates"],
        bonds=pubchem_qc_example["bond-connections"],
        bond_orders=pubchem_qc_example["bond-order"],
    )
    token_target = {k: pubchem_qc_example[k] for k in pubchem_qc.DEFAULT_TOKEN_TARGETS}
    mol_ref = add_atomic_properties(mol_ref, token_target)
    mol_ref = pubchem_qc.smear_hydrogen_targets(mol_ref, list(token_target.keys()))

    # Get Ref Smi Order, to remap atoms between mol and mol_ref
    Chem.MolToSmiles(mol_ref, canonical=False)
    ref_smi_order = [
        int(c) for c in mol_ref.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")
    ]

    # Reconstruct mol from prediction
    assert isinstance(out["target"], torch.Tensor) and isinstance(
        out["token_target"], torch.Tensor
    )
    mol = mol_from_prediction(
        torch.tensor(out["input_ids"]),
        y_seq=out["target"],
        y_token=out["token_target"],
        token_targets=pubchem_qc.DEFAULT_TOKEN_TARGETS,
        seq_targets=pubchem_qc.DEFAULT_SEQ_TARGETS,
        tokenizer=tokenizer,
        include_3d=include_3d,
    )
    assert isinstance(mol, Chem.Mol)
    conf = None
    conf_ref = None
    if include_3d:
        conf = mol.GetConformer()
        conf_ref = mol_ref.GetConformer()

    for idx, atom in enumerate(mol.GetAtoms()):
        ref_atom = mol_ref.GetAtomWithIdx(ref_smi_order[idx])
        target_props = {}
        ref_target_props = {"atomic-number": ref_atom.GetAtomicNum()}
        position = None
        position_ref = None
        logging.debug(atom.GetPropsAsDict())
        for k in pubchem_qc.DEFAULT_TOKEN_TARGETS:
            assert atom.GetAtomicNum() == ref_atom.GetAtomicNum()
            for target in [k, f"hs_{k}"]:
                assert atom.HasProp(target)  # Properties are defined everywhere
                if atom.HasProp(target):
                    target_props[target] = atom.GetDoubleProp(target)
                if ref_atom.HasProp(target):
                    ref_target_props[target] = ref_atom.GetDoubleProp(target)

        if include_3d:
            assert conf is not None and conf_ref is not None
            position = torch.tensor(conf.GetAtomPosition(atom.GetIdx()))
            position_ref = torch.tensor(conf_ref.GetAtomPosition(ref_atom.GetIdx()))
            assert position.isclose(position_ref, atol=1e-4, rtol=1e-4).all()

    s = pubchem_qc.serialize_molecule(mol)
    logging.debug(s)
    assert isinstance(s, dict)
    assert isinstance(json.dumps(s), str)


def test_distance_loss():
    # Shifts shouldn't increase the loss
    x = torch.rand(5, 32, 3)
    y = x + torch.rand(1, 1, 3)
    mask = torch.ones(5, 32, dtype=torch.bool)
    loss = distance_matrix_loss(x, y, mask)
    assert loss.isclose(torch.tensor(0.0))

    # Or rotations
    def rot(θ=1):
        return torch.tensor([[cos(θ), -sin(θ), 0], [sin(θ), cos(θ), 0], [0, 0, 1]])

    y = x @ rot().reshape(1, 3, 3)
    loss = distance_matrix_loss(x, y, mask)
    assert loss.isclose(torch.tensor(0.0))

    # Or random orthogonal transforms
    y = x @ torch.nn.init.orthogonal_(torch.ones(3, 3))
    loss = distance_matrix_loss(x, y, mask)
    assert loss.isclose(torch.tensor(0.0))

    # Skewing the input does
    y = x @ torch.rand(3, 3)
    loss = distance_matrix_loss(x, y, mask)
    assert not loss.isclose(torch.tensor(0.0))


@pytest.mark.parametrize("N", [32, 1, 2])
def test_mds_svd(N):
    coords = torch.rand(N, 3)
    D = torch.cdist(coords, coords)
    assert D.shape == (N, N)
    est_coords = pubchem_qc.mds_svd(D)
    assert est_coords.shape == (N, 3)
    D_est = torch.cdist(est_coords, est_coords)
    logging.info({"max_error": (D - D_est).max().item()})
    assert D.isclose(D_est, atol=5e-4).all()


@pytest.mark.parametrize("B,N", [(8, 32), (1, 1), (1, 8), (8, 1)])
def test_masked_mds_svd(B, N):
    mask = torch.rand(B, N) > 0.8
    mask_pw = mask.unsqueeze(2) & mask.unsqueeze(1)
    coords = torch.rand(B, N, 3)
    D = torch.cdist(coords, coords)
    D[~mask_pw] = 0
    assert D.shape == (B, N, N)

    est_coords = pubchem_qc.masked_mds_svd(D, mask)
    assert est_coords.shape == (B, N, 3)
    est_ref_coords = pubchem_qc.mds_svd(D[0])
    D_est = torch.cdist(est_coords, est_coords)
    D_est_ref = torch.cdist(est_ref_coords, est_ref_coords)

    logging.info(
        {
            "D": D[0],
            "D_est": D_est[0],
            "D_est_ref": D_est_ref,
            "max_error": (D - D_est).max().item(),
        }
    )
    assert D_est.isclose(D, atol=5e-4)[mask_pw].all()


def test_sparse_topo_distance():
    mol = Chem.MolFromSmiles("CC#N")
    atom_indixes = [0, 1, 3]
    s = pubchem_qc.sparse_topo_distance(mol, atom_indixes)
    assert s.shape == (4, 4, 2)
