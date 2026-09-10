from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection

from ..utils.ckpt import SaveConfigWithCkpts


def create_position_ids(
    input_ids: torch.Tensor, token_padding_idx: int, position_padding_idx: int
) -> torch.Tensor:
    """Create positions without changing a pretrained positional offset."""
    mask = input_ids.ne(token_padding_idx).to(dtype=torch.long)
    return mask.cumsum(dim=-1) * mask + position_padding_idx


def load_encoder(name_or_path: str, strict: bool = False):
    if Path(name_or_path).exists():
        return DeepSpeedMixin.load(name_or_path, strict=strict).get_encoder()
    else:
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            name_or_path,
            trust_remote_code=True,
        )


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


def record_summary_stats(logger, metrics: MetricCollection):
    if isinstance(logger, WandbLogger):
        define_metric = logger.experiment.define_metric
        for name, metric in metrics.items():
            if (
                not hasattr(metric, "higher_is_better")
                or metric.higher_is_better is None
            ):
                continue
            for phase in ["", "_epoch", "_step"]:
                define_metric(
                    name + phase,
                    summary="last,best",
                    goal="maximize" if metric.higher_is_better else "minimize",
                )


def record_loss_summary_stats(logger):
    if isinstance(logger, WandbLogger):
        define_metric = logger.experiment.define_metric
        for m in ["train/loss", "val/loss", "test/loss"]:
            for s in ["", "_step", "_epoch"]:
                define_metric(m + s, summary="last,best,min", goal="minimize")


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


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    x_masked = x * mask
    s = x_masked.sum(dim=-2)
    c = mask.sum(dim=-2).clamp(min=1)
    return s / c


def sparsity_weights(
    ds, column: list[str], eps=1e-6, max=None
) -> dict[str, torch.Tensor]:
    """Compute the relative sparsity of each channel in the target columns"""
    sparsity = {}
    for row in ds:
        for col in column:
            if col not in sparsity:
                sparsity[col] = torch.zeros_like(row[col], dtype=torch.float32)

            sparsity[col] += row[col]

    # Normalize
    for col in sparsity.keys():
        sparsity[col] = (sparsity[col].max() / (sparsity[col] + eps)).clamp(
            min=1, max=None
        )
        assert sparsity[col].min() >= 1

    return sparsity
