from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from ..utils.ckpt import SaveConfigWithCkpts


class DeepSpeedMixin:
    @staticmethod
    def load(checkpoint_dir, **kwargs):
        print(checkpoint_dir)
        return SaveConfigWithCkpts.load(checkpoint_dir, **kwargs)

    def load_state(self, checkpoint_dir):
        print("Loading state for checkpoint:", checkpoint_dir)
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        self.load_state_dict(state, strict=False, assign=True)

    def get_encoder(self):
        raise NotImplementedError


class LoggingMixin:
    def on_train_epoch_start(self) -> None:
        # Update the dataset's internal epoch counter

        self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)
        self.log(
            "train/dataloader_epoch",
            self.trainer.train_dataloader.dataset._epoch,
            rank_zero_only=True,
            sync_dist=True,
        )
        return super().on_train_epoch_start()


class CanSkip:
    def should_skip(self):
        """Return true if the model should skip this batch, the model
        should still perform a forward pass, but return `0 * loss`
        instead. Do not return `None` as unsupported by DeepSpeed
        """
        if hasattr(self, "skip_this_batch") and self.skip_this_batch:
            return True
        else:
            return False
