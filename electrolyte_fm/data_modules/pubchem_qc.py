import json
import logging
import traceback
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional
from enum import IntEnum, auto

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from rdkit import Chem

from electrolyte_fm.utils.tokenizer import load_tokenizer

from .utils import maybe_shard_dataset

DEFAULT_SEQ_TARGETS = [
    "total-energy",
    "charge",
    "dipole-moment",
    "alpha-homo",
    "alpha-lumo",
    "alpha-gap",
    "beta-homo",
    "beta-lumo",
    "beta-gap",
]

DEFAULT_TOKEN_TARGETS = ["partial-charge-mulliken", "partial-charge-lowdin"]


class PubChemQC(LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str = "smirk-cls",
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        randomize: bool = True,
        include_3d: bool = False,
        seq_targets: list[str] = DEFAULT_SEQ_TARGETS,
        token_targets: list[str] = DEFAULT_TOKEN_TARGETS,
        **kwargs,
    ):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.randomize = randomize
        self.include_3d = include_3d
        self.seq_targets = seq_targets
        self.token_targets = token_targets

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.save_hyperparameters()

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        self.__dataset = self._get_dataset()
        return self.__dataset

    def _get_dataset(self):
        return load_dataset(
            "arrow",
            name=self.path.name,
            data_files={
                "train": str(self.path.joinpath("train/*.arrow")),
                "validation": str(self.path.joinpath("validation/*.arrow")),
                "test": str(self.path.joinpath("test/*.arrow")),
            },
            streaming=True,
        )

    def prepare_data(self) -> None:
        self.dataset

    def setup(self, stage: str) -> None:
        ds = self.dataset
        if hasattr(self, "trainer"):
            ds = maybe_shard_dataset(self.trainer, ds)

        ds = ds.map(
            collate_partial_charges,
            batched=False,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "randomize": self.randomize,
                "include_3d": self.include_3d,
                "seq_targets": self.seq_targets,
                "token_targets": self.token_targets,
                "labeler": TokenLabeler(self.tokenizer),
            },
        )
        ds = ds.select_columns(
            [
                "input_ids",
                "attention_mask",
                "target",
                "target_mask",
                "token_target",
                "token_target_mask",
            ]
        )

        self.train_dataset: Dataset = ds["train"].shuffle(seed=42)
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        self.token_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")

    def collate_fn(self, batch):
        output = self.token_collator(
            [
                {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
                for x in batch
            ]
        )
        output["target"] = torch.stack([x["target"] for x in batch])
        output["target_mask"] = torch.stack([x["target_mask"] for x in batch])
        output["token_target"] = pad_sequence(
            [x["token_target"] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        output["token_target_mask"] = pad_sequence(
            [x["token_target_mask"] for x in batch],
            batch_first=True,
            padding_value=False,
        )
        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def collate_partial_charges(
    row: dict,
    tokenizer,
    seq_targets: list[str] = DEFAULT_SEQ_TARGETS,
    token_targets: list[str] = DEFAULT_TOKEN_TARGETS,
    randomize: bool = True,
    include_3d: bool = False,
    labeler: "TokenLabeler | None" = None,
):
    mol = construct_mol(
        atomic_numbers=row["atomic-numbers"],
        positions=row["atomic-coordinates"],
        bonds=row["bond-connections"],
        bond_orders=row["bond-order"],
    )
    token_target = {k: row[k] for k in token_targets}
    mol = add_atomic_properties(mol, token_target)

    # Smear hydrogen targets onto central atoms
    mol = smear_hydrogen_targets(mol, token_target.keys())

    # Annotate tokens with data
    # Log any errors before rethrowing
    out = None
    try:
        out = annotated_tokens(
            mol,
            targets=token_targets,
            tokenizer=tokenizer,
            doRandom=randomize,
            include_3d=include_3d,
            labeler=labeler,
        )
    except Exception:
        logging.error(
            "Error annotating `%s`: %s",
            json.dumps({k: v for k, v in row.items()}),
            traceback.format_exc(),
        )
        raise

    out["target"] = torch.tensor([row[k] for k in seq_targets])
    out["target_mask"] = torch.ones(out["target"].shape, dtype=bool)

    return out


def _best_effort_sanitize(mol: Chem.Mol):
    sanitize_ops = (
        Chem.rdmolops.SANITIZE_CLEANUPCHIRALITY | Chem.rdmolops.SANITIZE_SETAROMATICITY
    )
    out = Chem.rdmolops.SanitizeMol(mol, sanitize_ops, catchErrors=True)
    if out != 0:
        logging.info("failed to fully sanitize %s: %s", Chem.MolToSmiles(mol), out)


def construct_mol(
    atomic_numbers: list[int],
    positions: list[float],
    bonds: list[int],
    bond_orders: list[int],
) -> Chem.Mol:
    mol = Chem.EditableMol(Chem.Mol())

    for idx, atomic_number in enumerate(atomic_numbers):
        atom = Chem.Atom(atomic_number)
        atom.SetIntProp("atom_index", idx)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    assert len(bonds) == 2 * len(bond_orders)
    for idx, order in enumerate(bond_orders):
        sdx = bonds[2 * idx]
        edx = bonds[2 * idx + 1]
        order = get_bond_order(order)
        mol.AddBond(sdx, edx, order)

    # Add conformer and assign stereochemistry
    mol = mol.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx in range(len(atomic_numbers)):
        conf.SetAtomPosition(idx, positions[3 * idx : 3 * idx + 3])
    mol.AddConformer(conf)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    # Sanitize the molecule
    _best_effort_sanitize(mol)

    return mol


def get_bond_order(order: int) -> Chem.BondType:
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }
    return bond_types.get(order, Chem.BondType.UNSPECIFIED)


def add_atomic_properties(mol, properties: dict[str, list[float]]):
    n_atoms = mol.GetNumAtoms()
    for values in properties.values():
        assert len(values) == n_atoms

    for atom in mol.GetAtoms():
        for key, values in properties.items():
            idx = atom.GetIntProp("atom_index")
            atom.SetDoubleProp(key, values[idx])

    return mol


def smear_hydrogen_targets(mol_explicit: Chem.Mol, targets: list[str]) -> Chem.Mol:
    """
    Remove implicit hydrogen atoms from the molecule, accumulating their targets onto the
    attached atom

    mol_explicit: Molecule with all atoms modeled as individual atoms (no implicit hydrogen)
    - Each atom must be tagged with an `atom_index` int property for tracking
    - Each atom must have no implicit Hs
    - Each atom must have a value set for all targets

    Targets for removed hydrogen atoms will be accumulated into `hs_{target}` properties
    for the output molecules
    """

    # Annotate atoms for tracking
    atom_map = {}
    for atom in mol_explicit.GetAtoms():
        idx = atom.GetIntProp("atom_index")
        assert atom.GetNumImplicitHs() == 0
        atom_map[idx] = atom

    # Remove hydrogen atoms
    mol = Chem.rdmolops.RemoveHs(
        mol_explicit,
        implicitOnly=False,
        updateExplicitCount=True,
        sanitize=False,
    )
    _best_effort_sanitize(mol)

    retained_atoms = set()
    for atom in mol.GetAtoms():
        retained_atoms.add(atom.GetIntProp("atom_index"))

    assert retained_atoms.issubset(atom_map.keys())

    # Identify removed atoms
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIntProp("atom_index")
        h_targets = defaultdict(list)

        # Collect properties of the implicit hydrogens
        for neighbor in atom_map[atom_idx].GetNeighbors():
            ndx = neighbor.GetIntProp("atom_index")
            if ndx not in retained_atoms:
                assert neighbor.GetSymbol() == "H"
                for target in targets:
                    h_targets[target].append(neighbor.GetDoubleProp(target))

        # Record smeared targets
        has_hs_target = False
        for target in targets:
            if value := h_targets.get(target, None):
                atom.SetDoubleProp(f"hs_{target}", mean(value))
                has_hs_target = True
        atom.SetBoolProp("has_hs_target", has_hs_target)

    return mol


def atom_has_smi_hcount(atom: Chem.Atom) -> bool:
    if atom.GetSmarts()[0] != "[":
        return False
    return atom.GetNumExplicitHs() > 0


class SmiTokenType(IntEnum):
    Special = auto()
    Structure = auto()
    Bracket = auto()
    Element = auto()
    ExplicitHydrogen = auto()
    Unknown = auto()  # Really unlabeled

    def __format__(self, spec):
        return f"{self.name}"


class TokenLabeler:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.element_ids: list[int] = self.get_element_ids()
        self.token_map: dict[str, int] = self.get_token_map()
        self.all_special_ids: list[int] = tokenizer.all_special_ids

    def get_element_ids(self):
        is_element_regex = re.compile(r"^[A-Za-z][a-z]?")
        element_ids = []
        for token, id in self.tokenizer.get_vocab().items():
            if is_element_regex.match(token):
                element_ids.append(id)

        return element_ids

    def get_token_map(self):
        token_map = {}
        tokenizer = self.tokenizer
        for token in ["[", "]", "(", ")", "H"]:
            token_map[token] = tokenizer.convert_tokens_to_ids(token)
        return token_map

    def __call__(self, input_ids: list[int]):
        all_special_ids = self.all_special_ids
        token_map = self.token_map
        element_ids = self.element_ids
        hydrogen_id = token_map["H"]
        in_bracket = False
        seen_element = False
        for token_id in input_ids:
            if token_id in all_special_ids:
                yield SmiTokenType.Special
                continue

            # Bracketed atom
            if token_id == token_map["["]:
                in_bracket = True
                seen_element = False
                yield SmiTokenType.Bracket
            elif token_id == token_map["]"]:
                in_bracket = False
                yield SmiTokenType.Bracket
            elif token_id in (token_map["("], token_map[")"]):
                yield SmiTokenType.Structure

            elif token_id in element_ids:
                if in_bracket and seen_element and token_id == hydrogen_id:
                    yield SmiTokenType.ExplicitHydrogen
                else:
                    seen_element = True
                    yield SmiTokenType.Element
            else:
                yield SmiTokenType.Unknown


def smi_token_type(tokenizer, input_ids: list[int]):
    return TokenLabeler(tokenizer)(input_ids)


def annotated_tokens(
    mol: Chem.Mol,
    targets: list[str],
    tokenizer,
    tokenizer_kwargs: dict | None = None,
    doRandom: bool = True,
    include_3d: bool = False,
    labeler: TokenLabeler | None = None,
):
    """
    Given a rdkit.Chem.Mol with atom-level properties, return a tokenized SMILES encoding
    with annotated token-level targets.

    Atom-level properties must be defined for all atoms (including implicit Hs)

    Tokenizer should be smirk or smirk-cls

    """

    smi = Chem.MolToSmiles(mol, canonical=False, doRandom=doRandom)
    smi_order = [int(c) for c in mol.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")]
    token_out = tokenizer.encode_plus(smi, **(tokenizer_kwargs or {}))
    input_ids = token_out["input_ids"]
    # y = torch.zeros(len(input_ids), 2 * len(targets) + 3 * include_3d)
    # mask = torch.zeros(y.shape, dtype=torch.bool)
    n_features = 2 * len(targets) + 3 * include_3d
    y = []
    mask = []
    smi_atom_idx = 0

    conf = mol.GetConformer() if include_3d else None
    labeler = labeler or TokenLabeler(tokenizer)
    prior_atom = None
    for idx, token_type in enumerate(labeler(input_ids)):
        y_atom = []
        mask_atom = []
        if token_type == SmiTokenType.Element:
            atom = mol.GetAtomWithIdx(smi_order[smi_atom_idx])
            for tdx, target in enumerate(targets):
                y_atom.append(atom.GetDoubleProp(target))
                mask_atom.append(True)
                # y[idx, 2 * tdx] = atom.GetDoubleProp(target)
                # mask[idx, 2 * tdx] = True
                if atom.HasProp(f"hs_{target}"):
                    y_atom.append(atom.GetDoubleProp(f"hs_{target}"))
                    mask_atom.append(True)
                else:
                    y_atom.append(0)
                    mask_atom.append(False)
                    # y[idx, 2 * tdx + 1] = atom.GetDoubleProp(f"hs_{target}")
                    # mask[idx, 2 * tdx + 1] = True

            if include_3d:
                pos = conf.GetAtomPosition(smi_order[smi_atom_idx])
                y_atom.extend([pos.x, pos.y, pos.z])
                mask_atom.extend([True, True, True])
                # y[idx, -3:] = torch.tensor(pos)
                # mask[idx, -3:] = True

            smi_atom_idx += 1
            prior_atom = atom

        elif token_type == SmiTokenType.ExplicitHydrogen:
            # Set Hs target for the atom on the hcount's H
            atom = prior_atom
            assert atom is not None
            for tdx, target in enumerate(targets):
                y_atom.extend([0, atom.GetDoubleProp(f"hs_{target}")])
                mask_atom.extend([False, True])

            if include_3d:
                y_atom.extend([0] * 3)
                mask_atom.extend([False] * 3)

                # y[idx, 2 * tdx + 1] = atom.GetDoubleProp(f"hs_{target}")
                # mask[idx, 2 * tdx + 1] = True
        else:
            y_atom = [0] * n_features
            mask_atom = [False] * n_features

        assert len(y_atom) == n_features
        assert len(mask_atom) == n_features
        y.append(y_atom)
        mask.append(mask_atom)

    return {
        **token_out,
        "token_target": torch.tensor(y),
        "token_target_mask": torch.tensor(mask),
        "smi": smi,
    }


def mol_from_prediction(
    input_ids: torch.Tensor,
    tokenizer,
    y_seq: torch.Tensor | None = None,
    y_token: torch.Tensor | None = None,
    seq_targets: list[str] | None = None,
    token_targets: list[str] | None = None,
    include_3d: bool = False,
    labeler: TokenLabeler | None = None,
) -> Chem.Mol:
    # Construct molecule and smi -> atom mapping
    smi = tokenizer.decode(input_ids, skip_special_tokens=True)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.MolToSmiles(mol, canonical=False)
    smi_order = [int(c) for c in mol.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")]

    # Tag molecule properties
    if y_seq is not None:
        assert seq_targets is not None
        for k, v in zip(seq_targets, y_seq):
            mol.SetDoubleProp(k, v.item())

    if y_token is None:
        return mol
    assert token_targets is not None

    if include_3d:
        assert token_targets is not None
        assert y_token.shape[-1] == 2 * len(token_targets) + 3
    else:
        return mol

    # Annotate atoms from prediction
    conf = Chem.Conformer(mol.GetNumAtoms())
    smi_atom_idx = 0
    labeler = labeler or TokenLabeler(tokenizer)
    for idx, token_type in enumerate(labeler(input_ids.tolist())):
        if token_type == SmiTokenType.Element:
            atom_idx = smi_order[smi_atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            smi_atom_idx += 1

            if include_3d:
                conf.SetAtomPosition(atom_idx, y_token[idx, -3:].tolist())

            for tdx, target in enumerate(token_targets):
                atom.SetDoubleProp(target, y_token[idx, 2 * tdx].item())
                atom.SetDoubleProp(f"hs_{target}", y_token[idx, 2 * tdx + 1].item())

    mol.AddConformer(conf)

    return mol


def serialize_molecule(mol: Chem.Mol):
    _best_effort_sanitize(mol)

    def serialize_atom(mol, atom: Chem.Atom):
        out = {}
        out["atomic-number"] = int(atom.GetAtomicNum())
        out["hs-count"] = int(atom.GetTotalNumHs())
        out["isotope"] = int(atom.GetIsotope())
        if mol.GetNumConformers() > 0:
            out["position"] = [
                float(x) for x in mol.GetConformer().GetAtomPosition(atom.GetIdx())
            ]

        out["properties"] = {
            str(k): v
            for k, v in atom.GetPropsAsDict().items()
            if isinstance(v, (bool, int, float, str))
        }

        return out

    return {
        "smi": Chem.MolToSmiles(mol, canonical=True),
        "atoms": [serialize_atom(mol, atom) for atom in mol.GetAtoms()],
        "bonds": [
            {
                "start_index": bond.GetBeginAtomIdx(),
                "end_index": bond.GetEndAtomIdx(),
                "bond-order": bond.GetBondTypeAsDouble(),
                "bond-type": str(bond.GetBondType()),
                "properties": dict(**bond.GetPropsAsDict()),
            }
            for bond in mol.GetBonds()
        ],
        "properties": dict(**mol.GetPropsAsDict()),
    }
