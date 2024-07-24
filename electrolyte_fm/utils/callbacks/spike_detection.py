import os
import warnings
from typing import Any, Mapping, Union

import pytorch_lightning as pl
import torch
from lightning.fabric.utilities.spike import SpikeDetection as FabricSpikeDetection
from pytorch_lightning.callbacks import Callback

from ..ckpt import SaveConfigWithCkpts


class SpikeDetection(FabricSpikeDetection, Callback):

    def __init__(self):
        super().__init__()
        self.checkpoint_path = None

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
            loss = outputs.detach()
        elif isinstance(outputs, Mapping):
            loss = outputs["loss"].detach()
        else:
            raise TypeError(
                f"outputs have to be of type torch.Tensor or Mapping, got {type(outputs).__qualname__}"
            )

        if batch_idx == 0:
            self.running_mean.to(trainer.strategy.root_device)

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

    def _resolve_ckpt_dir(self, trainer: "pl.Trainer"):
        if self.checkpoint_path is None:
            checkpoint_path = None
            if trainer.is_global_zero:
                checkpoint_path = str(
                    SaveConfigWithCkpts.log_dir(trainer).joinpath("checkpoints")
                )
            self.checkpoint_path = trainer.strategy.broadcast(checkpoint_path, src=0)
        return self.checkpoint_path

    def _handle_spike(self, trainer: "pl.Trainer", batch_idx: int) -> None:

        checkpoint_path = self._resolve_ckpt_dir(trainer)

        # save current model as a checkpoint
        save_checkpoint_path = os.path.join(
            checkpoint_path, f"spike_step_{trainer.global_step}.ckpt"
        )
        trainer.save_checkpoint(save_checkpoint_path)

        # load model weights from last checkpoint
        last_checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
        trainer.model.exclude_batches.extend([batch_idx - 1, batch_idx])

        checkpoint = trainer.strategy.load_checkpoint(last_checkpoint_path)
        trainer.strategy.load_model_state_dict(
            checkpoint=checkpoint,
        )

        print(f"Resuming from checkpoint : {last_checkpoint_path}")
        print(f"Excluded batches : {trainer.model.exclude_batches}")

    def _is_spike(self, loss: torch.Tensor) -> bool:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # we might call compute more often than update
            # which is fine as long as the metric has
            # at least one internal value.
            running_val = self.running_mean.compute()
        curr_diff = loss - self.last_val

        if self.finite_only and not torch.isfinite(loss):
            return True

        if self._is_better(curr_diff):
            return False

        check_atol = bool(abs(running_val - loss) >= abs(self.atol))
        return check_atol
