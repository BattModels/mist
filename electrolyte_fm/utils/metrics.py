import torch
from torchmetrics import Metric
from torchmetrics.classification import AUROC, AveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

""" Target Value to indicate missing data """
IGNORE_INDEX = -100


class AvgMeanSquaredError(MeanSquaredError):
    """Computes the Average MSE of multiple output predictions"""

    def __init__(self, squared: bool = True, num_outputs: int = 1):
        super().__init__(squared=squared, num_outputs=num_outputs)

    def compute(self) -> torch.Tensor:
        return super().compute().mean()


def get_metric(name: str, task_type: str, output_size: int) -> Metric:
    if name == "auroc" and task_type == "binary_classification":
        return AUROC(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=100,
        )
    elif name == "avg-precision" and task_type == "binary_classification":
        return AveragePrecision(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=100,
        )
    elif name == "mae" and task_type == "regression":
        return MeanAbsoluteError()
    elif name == "avg-rmse" and task_type == "regression":
        return AvgMeanSquaredError(
            squared=True,
            num_outputs=output_size,
        )
    elif name == "r2" and task_type == "regression":
        return R2Score(num_outputs=output_size)
    else:
        raise ValueError(f"Unknown metric {name} for {task_type} tasks")


def masked_loss(
    lossfn,
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.FloatTensor,
) -> torch.Tensor:
    """Batch Averaged Loss, masking out unknown entries in y"""
    loss = lossfn(preds, targets.to(preds))
    return loss.masked_fill(mask, 0).sum() / mask.sum()


def masked_metric_forward(
    metrics: list[Metric],
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.BoolTensor,
) -> dict[str, torch.Tensor]:
    """Update metrics, masking out targets as needed"""
    out = {}
    targets = targets.masked_fill(mask, IGNORE_INDEX)
    for name, metric in metrics.items():
        out[name] = metric(preds, targets)

    return out
