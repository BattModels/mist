import json
from unittest import mock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from electrolyte_fm.utils.callbacks import SpikeDetection


@pytest.fixture()
def trainer():
    _trainer = Trainer(
        default_root_dir="/tmp",
        logger=None,
    )
    return _trainer


def test_resolve_ckpt_dir(trainer):
    callback = SpikeDetection()
    assert (
        callback._resolve_ckpt_dir(trainer)
        == "/tmp/lightning_logs/version_0/checkpoints"
    )


def test_is_spike():
    callback = SpikeDetection(atol=0.3)
    callback.running_mean.update(1.0)
    _loss = torch.tensor(1.4)
    assert callback._is_spike(_loss)
