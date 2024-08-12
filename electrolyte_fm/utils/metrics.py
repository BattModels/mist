from typing import Union, Dict, Optional

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.wrappers.classwise import ClasswiseWrapper
from torchmetrics.classification import AUROC, AveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

""" Target Value to indicate missing data """
IGNORE_INDEX = -100


class SafeR2Score(R2Score):
    def compute(self):
        if self.total < 2:
            return torch.tensor(float("nan"))
        return super().compute()


class OOVMetric(Metric):
    """Track metrics for OOV, Non-OOV and All tokenized strings"""

    def __init__(self, metric: Metric, unk_token_id: int) -> None:
        super().__init__()
        if not isinstance(metric, (Metric, MetricCollection)):
            raise ValueError(
                f"Expected metric to be a torchmetrics.Metric or torchmetrics.MetricCollection but got {metric}"
            )
        self.metrics = torch.nn.ModuleDict(
            {"oov": metric.clone(), "non_oov": metric.clone(), "all": metric.clone()},
        )
        self.unk_token_id = unk_token_id

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        input_ids: torch.IntTensor,
        is_oov: Optional[torch.BoolTensor] = None,
    ) -> None:
        assert input_ids.ndim == 2
        is_oov = (
            is_oov if is_oov is not None else (input_ids == self.unk_token_id).any(1)
        )
        self.metrics["all"].update(preds, targets)
        self.metrics["oov"].update(preds[is_oov], targets[is_oov])
        self.metrics["non_oov"].update(preds[~is_oov], targets[~is_oov])

    def compute(self):
        out = {}
        for k in self.metrics.keys():
            m = self.metrics[k].compute()
            if isinstance(m, Dict):
                out.update({mk + f"_{k}": mv for mk, mv in m.items()})
            else:
                out.update({k: m})
        return out

    def reset(self):
        for m in self.values():
            m.reset()

    def items(self):
        for k in self.metrics.keys():
            m = self.metrics[k]
            if isinstance(m, MetricCollection):
                for mk, mv in m.items():
                    yield (mk + f"_{k}", mv)
            else:
                yield k, m

    def keys(self):
        for k, _ in self.items():
            yield k

    def values(self):
        for _, v in self.items():
            yield v


def get_metric(name: str, task_type: str, output_size: int) -> Metric:
    if name == "auroc" and task_type == "binary":
        return AUROC(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=250,
        )
    elif name == "avg-precision" and task_type == "binary":
        return AveragePrecision(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=250,
        )
    elif name == "mae" and task_type == "regression":
        return MeanAbsoluteError()
    elif name == "rmse" and task_type == "regression":
        return MeanSquaredError(squared=True)
    elif name == "rmse" and task_type == "regression":
        return MeanSquaredError(squared=True)
    elif name == "r2" and task_type == "regression":
        return SafeR2Score(num_outputs=output_size)
    else:
        raise ValueError(f"Unknown metric {name} for {task_type} tasks")


def masked_loss(
    lossfn,
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.FloatTensor,
) -> torch.Tensor:
    """Batch Averaged Loss, masking out unknown entries in y"""
    if lossfn.reduction != "none":
        raise RuntimeError("Reduction must be 'none'")
    loss = lossfn(preds, targets.to(preds))
    return loss.masked_fill(mask, 0).sum() / mask.bitwise_not().sum()


def masked_metric_update(
    metrics: Metric,
    preds: torch.FloatTensor,
    targets: Union[torch.IntTensor, torch.FloatTensor],
    mask: torch.BoolTensor,
    *args,
):
    """Update metrics, masking out targets as needed"""
    targets = targets.masked_fill(mask, IGNORE_INDEX)
    metrics.update(preds, targets, *args)
