import lightning as L
from lightning.pytorch.callbacks import Callback


class GradientNormMonitor(Callback):
    """Custom callback in order to monitor and log the gradient norm."""

    def __init__(self) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()

    def on_after_backward(
        self, trainer: "L.pytorch.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are unscaled here
        # Avoid importing DeepSpeed at module import time: its compatibility
        # probe requires nvcc, which is absent on runtime-only GPU nodes.
        if "deepspeed" in trainer.strategy.__class__.__name__.lower():
            from deepspeed.utils import safe_get_full_grad

            gradients = (
                (name, safe_get_full_grad(parameter))
                for name, parameter in pl_module.model.named_parameters()
            )
        else:
            gradients = (
                (name, parameter.grad)
                for name, parameter in pl_module.model.named_parameters()
            )
        norms = {
            f"grad_norm/{name}": gradient.norm(2)
            for name, gradient in gradients
            if gradient is not None
        }
        pl_module.log_dict(norms, on_step=True, logger=True, sync_dist=True)
