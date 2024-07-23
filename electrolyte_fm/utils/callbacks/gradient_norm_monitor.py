import json
import os
import time
from tabnanny import check
from typing import Any, Mapping, Union

import pytorch_lightning as pl
from deepspeed.utils import safe_get_full_grad
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from typing_extensions import override

from ...models.model_utils import DeepSpeedMixin
from ..ckpt import SaveConfigWithCkpts


class GradientNormMonitor(Callback):
    """Custom callback in order to monitor the gradient norm and log to weights and biases."""

    def __init__(self) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()

    def on_after_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = {
            f"grad_norm/{n}": safe_get_full_grad(p).norm(2)
            for n, p in pl_module.model.named_parameters()
            if safe_get_full_grad(p) is not None
        }
        pl_module.log_dict(norms, on_step=True, logger=True, sync_dist=True)
