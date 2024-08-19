import json
from pathlib import Path
from typing import List
from itertools import product

import _jsonnet as jsonnet
import pytest
from pytorch_lightning import LightningDataModule
from datasets import DatasetDict, load_dataset

from electrolyte_fm.data_modules import MolNetDataModule, PropertyPredictionDataModule
from electrolyte_fm.data_modules.molnet_dataset import _URLS as MOLNET_URLS

MOLNET_CONFIG = Path(__file__).parent.parent.joinpath(
    "submit", "moleculenet_tasks.libsonnet"
)


def check_datamodule(dm: LightningDataModule, stage="fit", limit_batches=100):
    dm.prepare_data()
    dm.setup(stage)
    check_dataloader(dm.train_dataloader(), limit_batches)
    check_dataloader(dm.val_dataloader(), limit_batches)
    check_dataloader(dm.test_dataloader(), limit_batches)


def check_dataloader(dl, limit_batches, keys=["input_ids", "attention_mask"]):
    for idx, batch in enumerate(dl):
        for key in keys:
            assert key in batch
        if idx >= limit_batches:
            break


@pytest.mark.parametrize("name", MOLNET_URLS.keys())
def test_datamodule(name):
    task_config = json.loads(jsonnet.evaluate_file(str(MOLNET_CONFIG)))[name]
    dm = MolNetDataModule(
        name=name,
        target_columns=task_config["target_columns"],
        split=task_config["split"],
    )
    check_datamodule(dm)


@pytest.mark.parametrize("name", MOLNET_URLS.keys())
def test_prepare(name):
    task_config = json.loads(jsonnet.evaluate_file(str(MOLNET_CONFIG)))[name]
    dm = MolNetDataModule(
        name=name,
        target_columns=task_config["target_columns"],
        split=task_config["split"],
    )
    dm.prepare_data()
    assert isinstance(dm.dataset, DatasetDict)
    for split in ["train", "validation", "test"]:
        assert split in dm.dataset
        assert set(dm.dataset[split].column_names) > set(task_config["target_columns"])
        assert dm.smi_column in dm.dataset[split].column_names


def test_validate_molnet_config():
    for name, config in json.loads(jsonnet.evaluate_file(str(MOLNET_CONFIG))).items():
        assert name in MOLNET_URLS
        assert isinstance(config["task"], str)
        assert isinstance(config["split"], str)
        assert isinstance(config["target_columns"], List)
        assert isinstance(config["metrics"], List)
        assert isinstance(config["metrics"], List)


DATASET_SIZE = {
    "clintox": 1484,  # Manually confirmed, website lists 1478
    "hiv": 41127,
    "qm9": 133885,
    "esol": 1128,
    "freesolv": 642,
    "lipo": 4200,
    "muv": 93087,
    "bace": 1513,
    "bbbp": 2039,
    "sider": 1427,
    "tox21": 7831,
    "toxcast": 8575,
    "clintox": 1478,
}


@pytest.mark.parametrize(
    "dataset,split", product(DATASET_SIZE.keys(), ["scaffold", "random"])
)
def test_splits(dataset, split):
    dm = MolNetDataModule(name=dataset, split=split)
    dm.prepare_data()
    ds = dm.dataset
    ds_ref = load_dataset(
        "csv",
        name="clintox",
        data_files=[MOLNET_URLS["clintox"]],
    )

    # Check Lengths
    assert len(ds_ref) == 1
    assert len(ds_ref["train"]) == DATASET_SIZE[dataset]
    assert "train" in ds and "validation" in ds and "test" in ds
    assert (
        len(ds["train"]) + len(ds["validation"]) + len(ds["test"])
        == DATASET_SIZE[dataset]
    )

    # Check for overlap
    mol = {split: set(ds[split]["smiles"]) for split in ["train", "validation", "test"]}
    print(ds["train"].to_pandas())
    print(ds["validation"].to_pandas())
    print(ds["test"].to_pandas())
    assert len(mol["train"].intersection(mol["test"])) == 0
    assert len(mol["train"].intersection(mol["validation"])) == 0
    assert len(mol["test"].intersection(mol["validation"])) == 0
