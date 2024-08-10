import os
import signal
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel

import train
from electrolyte_fm.utils.callbacks import Requeue
from train import cli_main

from .test_dataset import fake_dataset


def test_requeue():
    with TemporaryDirectory() as root_dir:
        model = BoringModel()
        trainer = Trainer(
            callbacks=[Requeue()],
            max_steps=15,
            limit_val_batches=0,
            default_root_dir=root_dir,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        # Get the instantiated callback
        cb = trainer.callbacks[0]
        assert isinstance(cb, Requeue)
        cb.on_train_start = MagicMock(wraps=cb.on_train_start)
        cb.handle_requeue_signal = MagicMock(wraps=cb.handle_requeue_signal)

        trainer.fit(model, BoringDataModule())

        # Check state is correct after training with no signal
        assert cb.ckpt_path is not None and cb.ckpt_path.is_dir()
        assert cb.signal is not None
        assert cb.requeue_count == 0
        cb.on_train_start.assert_called()
        cb.handle_requeue_signal.assert_not_called()

        # Fake a signal
        cb.requeue = Mock(return_value=None)
        cb.handle_requeue_signal(cb.signal, None)
        assert cb.requeue_count == 1
        assert cb.ckpt_path.joinpath("requeue.ckpt").exists()
        cb.requeue.assert_called_once()

        # Resume
        trainer.fit(model, ckpt_path=cb.ckpt_path.joinpath("requeue.ckpt"))
        assert cb is trainer.callbacks[0]
        assert cb.requeue_count == 1


def test_cli():
    with TemporaryDirectory() as fake_data_dir:
        config = {
            "data": {
                "class_path": "electrolyte_fm.data_modules.RobertaDataSet",
                "init_args": {
                    "path": fake_data_dir,
                    "tokenizer": "ibm/MoLFormer-XL-both-10pct",
                },
            },
            "model": {
                "class_path": "electrolyte_fm.models.RoBERTa",
            },
            "trainer": {
                "devices": 1,
                "accelerator": "cpu",
                "callbacks": [
                    {"class_path": "electrolyte_fm.utils.callbacks.Requeue"},
                ],
            },
        }
        cli = cli_main(["--config", json.dumps(config)])
        cb = next(filter(lambda x: isinstance(x, Requeue), cli.trainer.callbacks), None)
        assert cb is not None


class SignalModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        if batch_idx > 50:
            os.kill(os.getpid(), signal.SIGUSR1)
        return super().training_step(batch, batch_idx)


class BoringBatchDataModule(BoringDataModule):
    def __init__(self):
        # ThroughputMonitor expects DataModule.batch_size to be defined
        self.batch_size = 64
        self.val_batch_size = self.batch_size
        super().__init__()


def test_signal(fake_dataset):
    with TemporaryDirectory() as root_dir:
        config = {
            "data": {"class_path": "test.test_requeue.BoringBatchDataModule"},
            "model": {"class_path": "test.test_requeue.SignalModel"},
            "trainer": {
                "devices": 1,
                "strategy": "auto",
                "accelerator": "cpu",
                "enable_progress_bar": True,
                "enable_model_summary": False,
                "default_root_dir": root_dir,
                "fast_dev_run": 100,
                "callbacks": [{"class_path": "electrolyte_fm.utils.callbacks.Requeue"}],
            },
        }
        p = subprocess.run(
            [
                sys.executable,
                Path(__file__).parent.parent.joinpath("train.py"),
                "fit",
                "--config",
                json.dumps(config),
            ],
            text=True,
            capture_output=True,
            check=False,  # Likely to error during requing
        )
        assert "Registered handler for SIGUSR1" in p.stderr
        assert "Requeuing using" in p.stderr
