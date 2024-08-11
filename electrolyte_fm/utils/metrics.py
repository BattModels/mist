from typing import Union, Dict

import torch
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.wrappers.classwise import ClasswiseWrapper
from torchmetrics.classification import AUROC, AveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

""" Target Value to indicate missing data """
IGNORE_INDEX = -100


class AvgMeanSquaredError(MeanSquaredError):
    """Computes the Average MSE of multiple output predictions"""

    def __init__(self, squared: bool = True, num_outputs: int = 1, **kwargs):
        super().__init__(squared=squared, num_outputs=num_outputs, **kwargs)

    def compute(self) -> torch.Tensor:
        return super().compute().mean()


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
        self, preds: torch.Tensor, targets: torch.Tensor, input_ids: torch.IntTensor
    ) -> None:
        assert input_ids.ndim == 2
        is_oov = (input_ids == self.unk_token_id).any(1)
        out = {}
        out["all"] = self.metrics["all"].forward(preds, targets)
        out["oov"] = self.metrics["oov"].forward(preds[is_oov], targets[is_oov])
        out["non_oov"] = self.metrics["non_oov"].forward(
            preds[~is_oov], targets[~is_oov]
        )

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
            thresholds=100,
        )
    elif name == "avg-precision" and task_type == "binary":
        return AveragePrecision(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=100,
        )
    elif name == "mae" and task_type == "regression":
        return MeanAbsoluteError()
    elif name == "rmse" and task_type == "regression":
        return MeanSquaredError(squared=True)
    elif name == "rmse" and task_type == "regression":
        return MeanSquaredError(squared=True)
    elif name == "r2" and task_type == "regression":
        return R2Score()
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


def masked_metric_forward(
    metrics: Union[dict[Metric], MetricCollection],
    preds: torch.FloatTensor,
    targets: Union[torch.IntTensor, torch.FloatTensor],
    mask: torch.BoolTensor,
    *args,
) -> dict[str, torch.Tensor]:
    """Update metrics, masking out targets as needed"""
    out = {}
    targets = targets.masked_fill(mask, IGNORE_INDEX)
    if isinstance(metrics, (MetricCollection, OOVMetric)):
        return metrics(preds, targets, *args)
    for name, metric in metrics.items():
        out[name] = metric(preds, targets, *args)

    return out


def record_summary_stats(logger, metrics: MetricCollection):
    if isinstance(logger, WandbLogger):
        define_metric = logger.experiment.define_metric
        for name, metric in metrics.items():
            define_metric(
                name + "_epoch",
                summary="last,best",
                goal="maximize" if metric.higher_is_better else "minimize",
            )
