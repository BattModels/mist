import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_relative_decay_with_warmup(
    current_step: int,
    *,
    num_training_steps: int,
    num_warmup_steps: int,
    rel_decay: float = 0.1,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    amp = 0.5 * (1 - rel_decay)
    offset = amp + rel_decay
    return max(
        rel_decay,
        amp * math.cos(2.0 * math.pi * float(num_cycles) * progress) + offset,
    )


class RelativeCosineWarmup(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int | str,
        rel_decay: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """
        Create a schedule with a learning rate that warms up to the initial lr set in the optimizer from zero,
        then decays following a cosine function to `rel_decay * initial_lr` after `num_training_steps`

        Args:
            optimizer: the optimizer for which to schedule the learning rate
            num_warmup_steps: the number of steps for the warm-up phase
            num_training_steps: the total number of training steps
            rel_decay: the fraction of the initial lr to decay to. Defaults to 0.1
            num_cycles: The number of waves in the cosine schedule
            last_epoch: index of the last epoch when resuming training

        If `num_warmup_steps == "beta"` sets num_warmup_steps to 2/(1 - β₂) per http://arxiv.org/abs/1910.04209.
        This only works if optimizer is in the Adam family
        """

        # Set num_warmup_steps from Adam's β₂ per http://arxiv.org/abs/1910.04209
        if isinstance(num_warmup_steps, str) and num_warmup_steps == "beta2":
            assert "betas" in optimizer.defaults
            beta2 = optimizer.defaults["betas"][1]
            num_warmup_steps = math.ceil(2 / (1 - beta2))

        assert isinstance(num_warmup_steps, int), f"Got {num_warmup_steps}, not an int"

        lr_lambda = partial(
            _get_cosine_relative_decay_with_warmup,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            rel_decay=rel_decay,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)
