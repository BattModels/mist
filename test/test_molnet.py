import json
from pathlib import Path
from typing import List

import _jsonnet as jsonnet
import pytest
from pytorch_lightning import LightningDataModule
from datasets import DatasetDict

from electrolyte_fm.data_modules import MolNetDataModule
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
