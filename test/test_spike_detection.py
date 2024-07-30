import json
from unittest.mock import MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel

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


def check_state(cb, **init_kwargs):
    state = cb.state_dict()
    cb_restored = cb.__class__(**init_kwargs)
    cb_restored.load_state_dict(state)
    assert cb_restored.state_dict() == state


def test_init_state():
    spike_cb = SpikeDetection()
    check_state(spike_cb)


def test_spiking():
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
            limit_val_batches=0,
            default_root_dir=ckpt_dir,
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
