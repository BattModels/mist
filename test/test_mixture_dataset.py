import csv
import json
import random
from copy import deepcopy

import pytest
import torch
from datasets import load_dataset
from smirk import SmirkTokenizerFast
from transformers import DataCollatorWithPadding

from electrolyte_fm.data_modules.mixture_dataset import (
    ComponentDataModule,
    encode_and_tokenize_mixture,
)
from electrolyte_fm.data_modules.utils import MolEncoding

SMILES = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CCCCCC",
    "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "COCCOC",
    "Clc1ccccc1C2(NC)CCCCC2=O",
    "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
]


@pytest.fixture(scope="module")
def tmp_dataset(tmp_path_factory):
    """Create a dummy CSV dataset with two targets."""
    root = tmp_path_factory.mktemp("mixture_data")
    random.seed(0)

    def write_split(split: str, n_rows: int):
        fn = root / f"{split}.csv"
        with fn.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "smi1",
                    "x1",
                    "smi2",
                    "x2",
                    "temperature",
                    "propA",
                    "propB",
                    "excess propA",
                    "excess propB",
                ]
            )
            for i in range(n_rows):
                x1 = random.random()
                writer.writerow(
                    [
                        random.choice(SMILES),
                        x1,
                        random.choice(SMILES),
                        1 - x1,
                        298.15,
                        random.random(),  # propA
                        random.random(),  # propB
                        random.random(),  # excess propA
                        random.random(),  # excess propB
                    ]
                )

    write_split("train", 32)
    write_split("validation", 32)
    write_split("test", 32)

    # Write to disk as an arrow dataset
    load_dataset(
        "csv",
        data_files={
            split: str(root / f"{split}.csv")
            for split in ["train", "validation", "test"]
        },
    ).save_to_disk(root)

    return root


@pytest.fixture(
    params=[
        {
            "randomize": randomize,
            "temperature_column": temperature_column,
            "include_encoding": include_encoding,
            "excess_columns": excess_columns,
            "num_workers": num_workers,
        }
        for randomize in [True, False]
        for temperature_column in ["temperature", None]
        for include_encoding in [True, False]
        for excess_columns in ["excess {:s}", ["excess propA", "excess propB"], None]
        for num_workers in [0, 2]
    ]
)
def datamodule(request, tmp_dataset):
    dm = ComponentDataModule(
        path=str(tmp_dataset),
        target_columns=["propA", "propB"],
        batch_size=8,
        num_workers=0,
        temperature_column=request.param["temperature_column"],
        tokenizer="smirk",
        randomize=request.param["randomize"],
        include_encoding=request.param["include_encoding"],
        excess_columns=request.param["excess_columns"],
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


def check_mixture_batch(dm: ComponentDataModule, batch: dict):
    assert batch["target"].shape == (dm.batch_size, len(dm.target_columns))
    assert batch["target"].dtype == torch.float32
    assert batch["target_mask"].shape == (dm.batch_size, len(dm.target_columns))
    assert batch["target_mask"].dtype == torch.bool
    assert batch["input_ids"].ndim == 3  # (B, n_components, seq_len)
    assert batch["input_ids"].shape[0] == dm.batch_size
    assert batch["input_ids"].shape[1] == dm.n_components
    assert batch["attention_mask"].ndim == 3
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["composition"].shape == (
        dm.batch_size,
        dm.n_components,
    )

    # Check that floating point numbers are fp32
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            assert v.dtype == torch.float32, f"{k} is of type {v.dtype} not float32"

    comp_sum = batch["composition"].sum(axis=1)
    assert comp_sum.shape == (dm.batch_size,)
    assert torch.allclose(comp_sum, torch.ones(dm.batch_size).to(comp_sum)), (
        f"Composition sums to 1: {comp_sum}"
    )

    if dm.temperature_column is not None:
        assert batch["temperature"].shape == (dm.batch_size,)
    else:
        assert "temperature" not in batch

    if dm.include_encoding:
        assert "compounds" in batch
        assert isinstance(batch["compounds"], list)
        assert len(batch["compounds"]) == dm.batch_size
        for compounds in batch["compounds"]:
            assert isinstance(compounds, tuple)
            assert len(compounds) == dm.n_components
            assert all(isinstance(smi, str) for smi in compounds)


@pytest.mark.parametrize(
    "compounds",
    [
        [("CO", "CC")],
        [("CO", "CC", "CCC"), ("C", "COC", "COCO")],
        [("C", "CC"), ("CO", "CC"), ("C", None)],  # Mix of order
        [tuple(random.choices(SMILES, k=2)) for n in range(8)],  # Binary
        [tuple(random.choices(SMILES, k=4)) for n in range(8)],  # 4-nary
    ],
)
def test_encode_and_tokenize_mixture(compounds):
    """Check that tokenization collation is ordered correctly"""
    assert all(isinstance(v, tuple) for v in compounds)
    print(json.dumps(compounds, indent=2))
    tokenizer = SmirkTokenizerFast()
    collator = DataCollatorWithPadding(tokenizer)
    out = encode_and_tokenize_mixture(
        deepcopy(compounds),
        tokenizer=tokenizer,
        encoding=MolEncoding.SMILES,
        randomize=False,
        token_collator=collator,
    )
    # Check output shapes and types
    B = len(compounds)
    N = len(compounds[0])
    S = max(
        len(tokenizer(compounds[bdx][idx] or "")["input_ids"])
        for bdx in range(B)
        for idx in range(N)
    )
    assert isinstance(out["input_ids"], torch.Tensor)
    assert isinstance(out["attention_mask"], torch.Tensor)
    assert out["input_ids"].shape == out["attention_mask"].shape
    assert out["input_ids"].shape == (B, N, S)

    # Check tokenization collation
    for bdx in range(B):
        for idx in range(N):
            smi = compounds[bdx][idx]
            assert isinstance(smi, str | None)
            tok = tokenizer(
                smi or "",
                padding="max_length",
                truncation=True,
                max_length=S,
            )
            assert isinstance(input_ids := tok["input_ids"], list)
            assert isinstance(attention_mask := tok["attention_mask"], list)
            assert len(input_ids) == S
            assert len(attention_mask) == S
            print("smi: ", smi)
            print("Got: ", out["input_ids"][bdx, idx, :])
            print("Expected: ", tok["input_ids"])
            assert out["input_ids"][bdx, idx, :].equal(torch.tensor(input_ids))
            assert out["attention_mask"][bdx, idx, :].equal(
                torch.tensor(attention_mask)
            )


def test_dataloader_iteration(datamodule):
    """Ensure val-loader yields batches without error and correct keys."""

    for split_loader in [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ]:
        for idx, batch in enumerate(split_loader):
            required_keys = {
                "composition",
                "input_ids",
                "attention_mask",
                "target",
                "target_mask",
            }
            assert required_keys <= batch.keys()
            check_mixture_batch(datamodule, batch)
            if idx >= 10:
                break
