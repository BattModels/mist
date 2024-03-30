""" Custom callbacks for benchmarking, adapted from GenSLM
    https://github.com/ramanathanlab/genslm/blob/71beb030df72010f5a4883a1f1a0b25bbafbe4a8/genslm/utils.py
"""

import time
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback


class ThroughputMonitor(Callback):
    """Custom callback in order to monitor the throughput and log to weights and biases."""

    def __init__(self) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()
        self.start_time = 0.0
        self.macro_batch_size = dict()

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.record_macro_batch_size("train", trainer.datamodule.batch_size, trainer)
        self.record_macro_batch_size("val", trainer.datamodule.val_batch_size, trainer)

    def record_macro_batch_size(
        self,
        stage: str,
        batch_size: int,
        trainer: "pl.Trainer",
    ):
        self.macro_batch_size[stage] = batch_size * trainer.world_size
        trainer.logger.log_hyperparams(
            {f"stats/{stage}_macro_batch_size": self.macro_batch_size[stage]},
        )

        # Configure summary metrics
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            logger = trainer.logger
            logger.experiment.define_metric(f"stats/{stage}_batch_time", summary="none")
            logger.experiment.define_metric(
                f"stats/{stage}_batch_throughput", summary="mean"
            )

    def start_batch_timer(self):
        self.start_time = time.perf_counter()

    def record_batch_perf(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ):
        batch_time = time.perf_counter() - self.start_time
        macro_batch = self.macro_batch_size.get(stage, 1)
        pl_module.log_dict(
            {
                f"stats/{stage}_batch_time": batch_time,
                f"stats/{stage}_batch_throughput": macro_batch / batch_time,
            },
            rank_zero_only=True,
            on_epoch=True,
        )

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.start_batch_timer()

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.start_batch_timer()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.record_batch_perf(trainer, pl_module, "train")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.record_batch_perf(trainer, pl_module, "val")
