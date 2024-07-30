import os
import warnings
import json
from typing import Any, Dict, Mapping, Union

import pytorch_lightning as pl
import torch
from lightning.fabric.utilities.spike import SpikeDetection as FabricSpikeDetection
from pytorch_lightning.callbacks import Callback

from ..ckpt import SaveConfigWithCkpts


class SpikeDetection(FabricSpikeDetection, Callback):
    def __init__(
        self,
        warmup: int = 200,
        atol: float = 0.3,
        finite_only: bool = True,
        checkpoint_spikes: bool = False,
    ):
        super().__init__(
            warmup=warmup, atol=atol, finite_only=finite_only, exclude_batches_path=None
        )
        self.checkpoint_path = None
        self.checkpoint_spikes = checkpoint_spikes

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        # Ensure metrics are moved to device before training starts
        self.running_mean.to(trainer.strategy.root_device)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        pl_module.skip_this_batch = batch_idx in self.bad_batches

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pl_module.skip_this_batch = False

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

        if self.exclude_batches_path is None:
            self.exclude_batches_path = SaveConfigWithCkpts.log_dir(trainer).joinpath(
                "skip_this_batch.json"
            )

        is_spike = bool(batch_idx >= self.warmup and self._is_spike(loss))
        trainer.strategy.barrier()

        # While spike-detection happens on a per-rank level, we need to fail all ranks if any rank detected a spike
        is_spike_global = trainer.strategy.reduce_boolean_decision(is_spike, all=False)

        if is_spike_global:
            self._handle_spike(trainer, batch_idx)
        else:
            is_finite_all = (
                self.finite_only
                or trainer.strategy.reduce_boolean_decision(
                    bool(torch.isfinite(loss).all()), all=True
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
        self.bad_batches.extend([batch_idx - 1, batch_idx])
        checkpoint_path = self._resolve_ckpt_dir(trainer)

        # Update batch_batches
        if trainer.is_global_zero:
            assert self.exclude_batches_path is not None
            with open(self.exclude_batches_path, "w") as f:
                json.dump(self.bad_batches, f, indent=4)

        # save current model as a checkpoint
        if self.checkpoint_spikes:
            trainer.save_checkpoint(
                os.path.join(checkpoint_path, f"spike_step_{trainer.global_step}.ckpt")
            )

        ckpt = os.path.join(checkpoint_path, "last.ckpt")
        assert os.path.exists(ckpt), f"no checkpoint to resume from: {ckpt}"
        checkpoint = trainer.strategy.load_checkpoint(ckpt)
        trainer.strategy.load_model_state_dict(
            checkpoint=checkpoint,
        )

        print(f"Resuming from checkpoint : {ckpt}")

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

    def state_dict(self):
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_spikes": self.checkpoint_spikes,
            **super(FabricSpikeDetection, self).state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.checkpoint_path = state_dict["checkpoint_path"]
        self.checkpoint_spikes = state_dict["checkpoint_spikes"]
        super(FabricSpikeDetection, self).load_state_dict(state_dict)
