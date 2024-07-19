import json
from pathlib import Path

import pytorch_lightning as pl

from ..utils.ckpt import SaveConfigWithCkpts
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

class DeepSpeedMixin:
    @staticmethod
    def load(checkpoint_dir, **kwargs):
        print(checkpoint_dir)
        return SaveConfigWithCkpts.load(checkpoint_dir, **kwargs)

        # config_path = config_path or Path(checkpoint_dir).parent.parent.joinpath(
        #     "config.json"
        # )
        # with open("config_path", "r") as fid:
        #     config = json.load(fid)
        #
        # cls.load(checkpoint_dir, config_path)
        #
    
    def load_state(self, checkpoint_dir):
        print("load_state", checkpoint_dir)
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        self.load_state_dict(state, strict=False, assign=True)

    def get_encoder(self):
        raise NotImplementedError


class LoggingMixin(pl.LightningModule):

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
