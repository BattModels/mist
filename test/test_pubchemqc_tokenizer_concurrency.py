"""Shared-tokenizer safety must not change the charge stream's order or labels."""

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

import pytest
import torch
from datasets import Dataset, DatasetDict
from smirk import SmirkTokenizerFast

from electrolyte_fm.data_modules.pubchem_qc import (
    DEFAULT_TOKEN_TARGETS,
    PubChemQC,
    TokenLabeler,
    collate_partial_charges,
)
from test_partial_charge_workflow import molecule_record


def exclusive_encode(monkeypatch, tokenizer):
    """Make the Rust mutable-borrow race deterministic without changing results."""
    encode = tokenizer.encode_plus
    borrowed = Lock()

    def checked(*args, **kwargs):
        assert borrowed.acquire(blocking=False), "concurrent tokenizer borrow"
        try:
            time.sleep(0.002)
            return encode(*args, **kwargs)
        finally:
            borrowed.release()

    monkeypatch.setattr(tokenizer, "encode_plus", checked)


def test_concurrent_charge_annotations_preserve_order_and_every_label(monkeypatch):
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    annotate = partial(
        collate_partial_charges,
        tokenizer=tokenizer,
        token_targets=DEFAULT_TOKEN_TARGETS,
        seq_targets=["charge"],
        randomize=False,
        labeler=TokenLabeler(tokenizer),
        smear_hydrogens=True,
    )
    rows = [molecule_record(smi) for smi in ("O", "CO", "CCO", "CCN") * 16]
    for index, row in enumerate(rows):
        row["charge"] = float(index)
    expected = [annotate(row) for row in rows]
    exclusive_encode(monkeypatch, tokenizer)
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(annotate, rows))
    assert [row["target"].item() for row in actual] == list(range(len(rows)))
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.keys() == reference.keys()
        for key in observed:
            if torch.is_tensor(reference[key]):
                assert torch.equal(observed[key], reference[key]), key
            else:
                assert observed[key] == reference[key], key


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_parallel_arrow_mapping_preserves_existing_split_order(
    tmp_path, monkeypatch, split
):
    rows = [molecule_record(smi) for smi in ("O", "CO", "CCO", "CCN") * 8]
    for index, row in enumerate(rows):
        row["charge"] = float(index)
    dataset = tmp_path / "data"
    DatasetDict(
        {name: Dataset.from_list(rows) for name in ("train", "validation", "test")}
    ).save_to_disk(
        dataset, num_shards={name: 4 for name in ("train", "validation", "test")}
    )
    data = PubChemQC(
        str(dataset),
        randomize=False,
        evaluation_randomize=False,
        seq_targets=["charge"],
        smear_hydrogens=True,
    )
    raw = data.dataset[split]
    if split == "train":
        raw = raw.shuffle(seed=42)
    expected = [row["charge"] for row in raw]
    exclusive_encode(monkeypatch, data.tokenizer)
    data.setup("fit")
    mapped = getattr(
        data,
        {"train": "train_dataset", "validation": "val_dataset", "test": "test_dataset"}[
            split
        ],
    )
    assert [row["target"].item() for row in mapped] == expected
