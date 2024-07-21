""" Custom callbacks for benchmarking, adapted from GenSLM
    https://github.com/ramanathanlab/genslm/blob/71beb030df72010f5a4883a1f1a0b25bbafbe4a8/genslm/utils.py
"""

import json
import os
from tabnanny import check
import time
from typing import Any, Mapping, Union

import pytorch_lightning as pl
import torch
from lightning.fabric.utilities.spike import SpikeDetection as FabricSpikeDetection
from pytorch_lightning.callbacks import Callback, Checkpoint
from pytorch_lightning.loggers import WandbLogger
from typing_extensions import override

from ..models.model_utils import DeepSpeedMixin
from .ckpt import SaveConfigWithCkpts
import warnings

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
        if trainer.is_global_zero:
            self.record_macro_batch_size(
                "train", trainer.datamodule.batch_size, trainer
            )
            self.record_macro_batch_size(
                "val", trainer.datamodule.val_batch_size, trainer
            )

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
            sync_dist=False,
        )

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.is_global_zero:
            self.start_batch_timer()

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.is_global_zero:
            self.start_batch_timer()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.is_global_zero:
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
        if trainer.is_global_zero:
            self.record_batch_perf(trainer, pl_module, "val")


class SpikeDetection(FabricSpikeDetection, Callback):

    @torch.no_grad()
    def on_train_batch_end(  # type: ignore
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        if isinstance(outputs, torch.Tensor):
            print(f"device: {outputs.device}")
            loss = outputs.detach()
        elif isinstance(outputs, Mapping):
            print(f"device: {outputs['loss'].device}")
            loss = outputs["loss"].detach()
        else:
            raise TypeError(
                f"outputs have to be of type torch.Tensor or Mapping, got {type(outputs).__qualname__}"
            )

        if self.exclude_batches_path is None:
            self.exclude_batches_path = os.path.join(
                trainer.default_root_dir, "skip_batches.json"
            )

        if batch_idx == 0:
            self.running_mean.to(trainer.strategy.root_device)

        
        print(f"global_step : {trainer.global_step}")
        print(f"batch_idx : {batch_idx}")
        print(f"running_val : {self.running_mean.compute()}")
        print(f"loss : {loss}")

        if self.exclude_batches_path is None:
            self.exclude_batches_path = os.getcwd()

        if not str(self.exclude_batches_path).endswith(".json"):
            self.exclude_batches_path = os.path.join(
                self.exclude_batches_path, "skip_batches.json"
            )

        is_spike = bool(batch_idx >= self.warmup and self._is_spike(loss))
        trainer.strategy.barrier()
        is_spike_global = trainer.strategy.reduce_boolean_decision(is_spike, all=True)

        if is_spike_global:
            self._handle_spike(trainer, batch_idx)
        else:
            is_finite_all = (
                self.finite_only
                or trainer.strategy.reduce_boolean_decision(
                    bool(torch.isfinite(loss).all()), all=False
                )
            )
            if is_finite_all:
                self._update_stats(loss)

    def __resolve_ckpt_dir(self, trainer: "pl.Trainer"):
                
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            checkpoint_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            checkpoint_path = os.path.join(trainer.default_root_dir, "checkpoints")

        ckpt_path = trainer.strategy.broadcast(checkpoint_path, src=0)
        return ckpt_path

    def _handle_spike(self, trainer: "pl.Trainer", batch_idx: int) -> None:

        trainer.model.exclude_batches.extend([batch_idx - 1, batch_idx])

        checkpoint_path = self.__resolve_ckpt_dir(trainer)
        last_checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
        print(f"Resuming from checkpoint : {last_checkpoint_path}")
        print(f"Excluded batches : {trainer.model.exclude_batches}")
        checkpoint =  trainer.strategy.load_checkpoint(last_checkpoint_path)
        trainer.strategy.load_model_state_dict(
            checkpoint = checkpoint,
            )
    
    def _is_spike(self, loss: torch.Tensor) -> bool:
        # we might call compute more often than update which is fine as long as the
        # metric has at least one internal value.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            running_val = self.running_mean.compute()
        curr_diff = loss - self.last_val
        

        if self.finite_only and not torch.isfinite(loss):
            return True

        if self._is_better(curr_diff):
            return False

        check_atol = bool(abs(running_val - loss) >= abs(self.atol))
        # check_rtol = bool(abs(running_val - loss) >= abs(self.rtol * loss)
        return check_atol # and check_rtol
    
        # print(f"load_path : {load_path}")
        # print(f"client_state : {client_state}")
        # trainer.model = loaded_checkpoint
        # trainer.model.load_state_dict(loaded_checkpoint)
        # trainer.strategy.barrier()
        # trainer.model.load_state(last_checkpoint_path)
        # trainer.model.to(trainer.strategy.root_device)
