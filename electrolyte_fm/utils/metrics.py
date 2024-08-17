from typing import Union, Dict, Optional

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.wrappers.classwise import ClasswiseWrapper
from torchmetrics.classification import (
    AUROC,
    AveragePrecision,
    Accuracy,
    BinaryStatScores,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

""" Target Value to indicate missing data """
IGNORE_INDEX = -100


class SafeR2Score(R2Score):
    def compute(self):
        if self.total < 2:
            return torch.tensor(
                float("nan"),
                device=self.total.device,
                dtype=self.sum_error.dtype,
            )
        return super().compute()


class BinaryDictStatScores(BinaryStatScores):
    def __init__(self, name, **kwargs):
        self.name = name
        if multidim_average := kwargs.pop("multidim_average", None):
            if multidim_average != "global":
                raise ValueError("multidim_average must be global")
        super().__init__(multidim_average="global", **kwargs)

    def update(self, preds: torch.FloatTensor, targets: torch.IntTensor) -> None:
        # StatScores doesn't support bf16
        super().update(preds.float(), targets.float())

    def compute(self) -> torch.Tensor:
        out = super().compute()
        return {
            "tp": out[0],
            "fp": out[1],
            "tn": out[2],
            "fn": out[3],
            "sup": out[4],
        }[self.name]


class HotellingTwoSample(Metric):
    def __init__(self, num_outputs: int, unk_token_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.unk_token_id = unk_token_id
        self.add_state("residual_oov", list(), dist_reduce_fx="cat")
        self.add_state("residual_non_oov", list(), dist_reduce_fx="cat")

    def update(self, preds, targets, input_ids, is_oov=None) -> None:
        is_oov = (
            is_oov if is_oov is not None else (input_ids == self.unk_token_id).any(1)
        )
        residual = preds - targets
        self.residual_oov.append(residual[is_oov])
        self.residual_non_oov.append(residual[~is_oov])

    def compute(self) -> torch.Tensor:
        # Collate residuals
        oov = self.residual_oov
        non_oov = self.residual_non_oov
        if oov.size(0) < 2 or non_oov.size(0) < 2:
            return dict()
        p = non_oov.size(1)
        assert (
            oov.ndim == 2 and non_oov.ndim == 2 and oov.size(1) == non_oov.size(1) == p
        )

        # Pooled Covariance matrix
        n_oov = oov.size(0)
        n_non_oov = non_oov.size(0)
        df = n_oov + n_non_oov - 2
        sigma = (n_oov - 1) * oov.T.cov() + (n_non_oov - 1) * non_oov.T.cov()
        sigma /= df
        assert sigma.ndim == 2
        assert sigma.size(0) == sigma.size(1) == p

        # Hotelling's T
        t = (n_oov * n_non_oov) / (n_oov + n_non_oov)
        avg_diff = oov.mean(0) - non_oov.mean(0)
        t *= avg_diff.dot(torch.linalg.solve(sigma, avg_diff))

        # Rescale to the F-distribution
        d2 = n_oov + n_non_oov - p - 1
        t_fdit = t * d2 / (df * p)
        return {
            "t2": t,
            "t_fdist": t_fdit,
            "df": df,
            "p": p,
            "d2": d2,
            "oov_rmse": oov.mean(0).norm(2),
            "non_oov_rmse": non_oov.mean().norm(2),
        }


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
        if is_oov.any():
            self.metrics["oov"].update(preds[is_oov], targets[is_oov])
        if not is_oov.all():
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


def get_metric(name: str, task_type: str, output_size: Optional[int] = None) -> Metric:
    if name == "auroc" and task_type == "binary":
        return AUROC(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=500,
        )
    elif name == "avg-precision" and task_type == "binary":
        return AveragePrecision(
            task="binary",
            ignore_index=IGNORE_INDEX,
            thresholds=250,
        )
    elif name == "crosstab" and task_type == "binary":
        return MetricCollection(
            {
                name: BinaryDictStatScores(name, ignore_index=IGNORE_INDEX)
                for name in ["tp", "tn", "fp", "fn", "sup"]
            }
        )
    elif name == "mae" and task_type == "regression":
        return MeanAbsoluteError()
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
