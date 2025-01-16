import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel

from electrolyte_fm.utils.callbacks import SpikeDetection


class SpikingModel(BoringModel):
    def __init__(self, spiking_batch=10, spike_value=10):
        self.spike_value = spike_value
        self.spiking_batch = spiking_batch
        super().__init__()

    def training_step(self, batch, batch_idx: int):
        outputs = super().training_step(batch, batch_idx)
        if batch_idx >= self.spiking_batch:
            outputs["loss"] *= self.spike_value
        return outputs

    def on_validation_start(self):
        # skip_this_batch should be false during validation
        if hasattr(self, "skip_this_batch"):
            assert not self.skip_this_batch

    def on_test_start(self):
        # skip_this_batch should be false during testing
        if hasattr(self, "skip_this_batch"):
            assert not self.skip_this_batch


def check_state(cb, **init_kwargs):
    state = cb.state_dict()
    cb_restored = cb.__class__(**init_kwargs)
    cb_restored.load_state_dict(state)
    assert cb_restored.state_dict() == state


def test_init_state():
    spike_cb = SpikeDetection()
    check_state(spike_cb)


def test_spiking(caplog):
    model = SpikingModel()
    data = BoringDataModule()
    spike_cb = SpikeDetection(warmup=0)
    with TemporaryDirectory() as ckpt_dir:
        trainer = Trainer(
            callbacks=[
                ModelCheckpoint(every_n_train_steps=1, save_last="link", verbose=True),
                spike_cb,
            ],
            max_steps=15,
            limit_val_batches=2,
            limit_test_batches=2,
            default_root_dir=ckpt_dir,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Setup trainer
        spike_cb = trainer.callbacks[0]
        spike_cb.running_mean.to = MagicMock(wrap=spike_cb.running_mean.to)
        assert isinstance(spike_cb, SpikeDetection)
        check_state(spike_cb)
        assert trainer.callbacks[0] is spike_cb
        assert not hasattr(model, "skip_this_batch")

        with caplog.at_level(logging.INFO):
            trainer.fit(model, data)

        # Check post fit state
        check_state(spike_cb)
        assert 10 in spike_cb.bad_batches
        assert (
            9 in spike_cb.bad_batches
        )  # Batch before the bad batch should also be skipped
        assert spike_cb.checkpoint_path is not None
        assert hasattr(model, "skip_this_batch") and not model.skip_this_batch
        assert Path(trainer.checkpoint_callback.last_model_path).parent == Path(
            spike_cb.checkpoint_path
        )
        spike_cb.running_mean.to.assert_called_once()

        # Check that resuming was logged
        found = False
        for record in caplog.records:
            if record.levelno != logging.INFO:
                continue
            if record.message.startswith("Spike detected"):
                found = True
                break
        assert found

        # Check that bad batches were recorded
        assert Path(spike_cb.exclude_batches_path).is_file()
        with open(spike_cb.exclude_batches_path, "r") as fid:
            bad_batches = json.load(fid)
        assert bad_batches == spike_cb.bad_batches

        # Resume, check that running_mean is moved to device
        # should only be called once
        spike_cb.running_mean.to.configure_mock(called=False, call_count=0)
        trainer.fit(model, ckpt_path=trainer.checkpoint_callback.last_model_path)
        spike_cb.running_mean.to.assert_called_once()


def test_is_spike():
    callback = SpikeDetection(atol=0.3)
    callback.running_mean.update(1.0)
    large_loss = torch.tensor(1.4)
    assert callback._is_spike(large_loss)
