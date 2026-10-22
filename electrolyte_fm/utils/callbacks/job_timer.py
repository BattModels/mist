"""Apply the wall-time limit separately to each resumed training job."""

from pathlib import Path

from lightning.pytorch.callbacks import Timer


class JobTimer(Timer):
    @property
    def state_key(self):
        # Accept checkpoints made with Lightning's original Timer as well.
        return "Timer"

    def load_state_dict(self, state_dict):
        # Model, optimizer and step state resume; the new allocation gets its
        # full wall-time budget instead of inheriting an expired timer.
        pass

    def on_fit_end(self, trainer, pl_module):
        checkpoint = trainer.checkpoint_callback
        if checkpoint is not None and checkpoint.save_last:
            # The last validation checkpoint can predate the final training step.
            # Save before strategy teardown so distributed optimizer state is
            # still available, without changing the validation-selected model.
            last = Path(checkpoint.dirpath) / "last.ckpt"
            checkpoint.last_model_path = str(last)
            trainer.save_checkpoint(last)
