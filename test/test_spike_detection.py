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
            outputs["loss"] * self.spike_value
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
        spike_cb = trainer.callbacks[0]
        assert isinstance(spike_cb, SpikeDetection)
        check_state(spike_cb)
        trainer.fit(model, data)
        check_state(spike_cb)
        assert 10 in spike_cb.bad_batches
        assert spike_cb.checkpoint_path is not None
        assert Path(trainer.checkpoint_callback.last_model_path).parent == Path(
            spike_cb.checkpoint_path
        )


def test_is_spike():
    callback = SpikeDetection(atol=0.3)
    callback.running_mean.update(1.0)
    large_loss = torch.tensor(1.4)
    assert callback._is_spike(large_loss)
