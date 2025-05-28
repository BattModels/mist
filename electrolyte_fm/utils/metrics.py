from typing import Any, Dict, Literal, Optional, Union

import torch
from torchmetrics import Metric
from torchmetrics import MetricCollection as TmMetricCollection
from torchmetrics.classification import (
    AUROC,
    AveragePrecision,
    BinaryStatScores,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)
from torchmetrics.wrappers import BootStrapper
from torchmetrics.wrappers.classwise import ClasswiseWrapper as TmClasswiseWrapper

""" Target Value to indicate missing data """
IGNORE_INDEX = -100


class SafeR2Score(R2Score):
    def __init__(self, *args, num_outputs=None, **kwargs):
        self.num_outputs = num_outputs
        super().__init__(*args, **kwargs)

    def compute(self):
        if self.total < 2:
            if self.num_outputs is not None:
                x = [float("nan") for _ in range(self.num_outputs)]
            else:
                x = float("nan")

            return torch.tensor(
                x,
                device=self.total.device,
                dtype=self.sum_error.dtype,
            )
        return super().compute()


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
        if isinstance(self.residual_oov, list):
            oov = torch.cat(self.residual_oov)
        else:
            oov = self.residual_oov
        if isinstance(self.residual_non_oov, list):
            non_oov = torch.cat(self.residual_non_oov)
        else:
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


class ClasswiseWrapper(TmClasswiseWrapper):
    def _convert_output(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Override the default _convert_output to support wrapping a single dict-returning metric"""
        if not isinstance(x, dict):
            return self._convert_output_tensor(x)

        out = {}
        for k, v in x.items():
            co = self._convert_output_tensor(v)
            for ck, cv in co.items():
                out[f"{ck}_{k}"] = cv

        return out

    def _convert_output_tensor(self, x: torch.Tensor) -> Dict[str, Any]:
        """Override the default all allow for blank prefixes and postfixes"""
        if self._prefix is None and self._postfix is None:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix if self._prefix is not None else ""
            postfix = self._postfix if self._postfix is not None else ""

        # Handle singletons
        if x.ndim == 0:
            assert self.labels is None or len(self.labels) == 1, f"{self}: {x}"
            x = [x]

        if self.labels is None:
            return {f"{prefix}{i}{postfix}": val for i, val in enumerate(x)}
        return {f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)}


class MetricCollection(TmMetricCollection):
    def _compute_and_reduce(
        self, method_name: Literal["compute", "forward"], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        result = {}
        """ Overrides the default implementation of MetricCollection._compute_and_reduce to support wrapping a single Bootstrapping metric"""
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute()
            elif method_name == "forward":
                res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(
                    f"method_name should be either 'compute' or 'forward', but got {method_name}"
                )
            result[k] = res

        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    # Strip prefixes and postfixes
                    stripped_k = k.replace(getattr(m, "prefix", ""), "")
                    stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                    key = f"{stripped_k}_{key}"

                    if getattr(m, "_from_collection", None) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", None) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}


def normalize_name(channel: str) -> str:
    return channel.replace("_", "-").replace("/", "--").strip()


def get_metrics(
    metrics: list[str],
    task: str,
    num_outputs: int = 1,
    target_channels: Optional[list[str]] = None,
    **kwargs,
) -> MetricCollection:
    mc = {}
    if target_channels is not None:
        target_channels = [normalize_name(c) for c in target_channels]
    for metric in metrics:
        if metric.endswith("-channel"):
            key = metric.replace("-", "_")
            mc[key] = get_metric(
                metric, task, num_outputs=num_outputs, target_channels=target_channels
            )
        else:
            mc[metric] = get_metric(metric, task)

    return MetricCollection(mc)


def get_metric(name: str, task_type: str, **kwargs) -> Metric:
    if name == "auroc" and task_type == "binary":
        if (num_labels := kwargs.pop("num_outputs", None)) and num_labels > 1:
            kwargs["task"] = "multilabel"
            kwargs["num_labels"] = num_labels
            kwargs["average"] = "none"
        else:
            kwargs["task"] = "binary"

        m = AUROC(
            ignore_index=IGNORE_INDEX,
            thresholds=500,
            **kwargs,
        )
    elif name == "avg-precision" and task_type == "binary":
        if (num_labels := kwargs.pop("num_outputs", None)) and num_labels > 1:
            kwargs["task"] = "multilabel"
            kwargs["num_labels"] = num_labels
            kwargs["average"] = "none"
        else:
            kwargs["task"] = "binary"

        m = AveragePrecision(
            ignore_index=IGNORE_INDEX,
            thresholds=250,
            **kwargs,
        )
    elif name == "crosstab" and task_type == "binary":
        m = ClasswiseWrapper(
            BinaryStatScores(ignore_index=IGNORE_INDEX, **kwargs),
            labels=["tp", "tn", "fp", "fn", "sup"],
            prefix="",
        )
    elif name == "mae" and task_type == "regression":
        m = MeanAbsoluteError(**kwargs)
    elif name == "mape" and task_type == "regression":
        m = MeanAbsolutePercentageError(**kwargs)
    elif name == "rmse" and task_type == "regression":
        m = MeanSquaredError(squared=False, **kwargs)
    elif name == "pearson" and task_type == "regression":
        m = PearsonCorrCoef(**kwargs)
    elif name == "r2" and task_type == "regression":
        multioutput = (
            "uniform_average"
            if kwargs.get("num_outputs", None) is None
            else "raw_values"
        )
        m = SafeR2Score(multioutput=multioutput, **kwargs)

    elif name.endswith("-channel"):
        target_channels = kwargs.pop("target_channels", None)
        if target_channels is None:
            assert kwargs.get("num_outputs", None) is not None
        else:
            kwargs["num_outputs"] = kwargs.get("num_outputs", len(target_channels))
            assert len(target_channels) == kwargs["num_outputs"]
            assert kwargs["num_outputs"] >= 1

        m = ClasswiseWrapper(
            get_metric(
                name.replace("-channel", ""),
                task_type,
                **kwargs,
            ),
            labels=target_channels,
            prefix="",
        )
    else:
        raise ValueError(f"Unknown metric {name} for {task_type} tasks")

    m.name = name

    return m


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
    input_ids: Optional[torch.IntTensor] = None,
    is_oov: Optional[torch.BoolTensor] = None,
    int_cast: bool = False,
):
    """Update metrics, masking out targets as needed"""
    targets = targets.masked_fill(~(mask.to(dtype=bool)), IGNORE_INDEX)
    if int_cast:
        targets = targets.int()
    if isinstance(metrics, OOVMetric):
        assert input_ids is not None
        metrics.update(preds, targets, input_ids, is_oov)
    else:
        metrics.update(preds, targets)


class TokenCounter(Metric):
    """Count number of tokens seen by model during training."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "masked_tokens",
            torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
            persistent=True,
        )
        self.add_state(
            "total_tokens",
            torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
            persistent=True,
        )

    def update(
        self,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.masked_tokens += torch.sum(labels != -100)
        self.total_tokens += attention_mask.count_nonzero()

    def compute(self):
        out = {
            "masked_tokens": self.masked_tokens,
            "total_tokens": self.total_tokens,
        }
        return out


def bootstrap_collection(metrics: MetricCollection, **kwargs) -> MetricCollection:
    """Apply Bootstrapping to all supported metrics in a MetricCollection"""
    mc = {}
    for k, v in metrics.items():
        if isinstance(v, ClasswiseWrapper):
            if k in ["crosstab"]:
                mc[k] = v  # Don't bootstrap unsupported metrics
            else:
                mc[k] = ClasswiseWrapper(
                    BootStrapper(v.metric, **kwargs),
                    labels=v.labels,
                    prefix=v._prefix,
                    postfix=v._postfix,
                )
        else:
            mc[k] = BootStrapper(v, **kwargs)

    return MetricCollection(mc)


class OrthoProcrustes(Metric):
    def __init__(self, reduction="mean", **kwargs):
        super.__init__(**kwargs)
        self.reduction = reduction
        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self):
        return self.distance / self.total

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        dists = self.procrustes_disparity(preds, targets)
        if self.reduction == "mean":
            self.distance += dists.mean()
        elif self.reduction == "rmsd":
            self.distance += dists.square().mean().sqrt()
        else:
            self.reduction = dists.sum()

        self.total += preds.size(0)

    @torch.cuda.nvtx.range("OrthoProcrustes.procrustes_disparity")
    @staticmethod
    def procrustes_disparity(preds: torch.Tensor, targets: torch.Tensor):
        # Zero centroids
        preds = preds - preds.mean(-2, keepdim=True)
        targets = targets - targets.mean(-2, keepdim=True)
        dtype = preds.dtype

        # Rotate targets to align with preds
        R = OrthoProcrustes.procrustes_alignment(targets, preds)
        targets = (targets @ R).to(dtype=dtype).detach()

        # Atom-wise distances
        return (preds - targets).pow(2).sum(-1).sqrt()

    @staticmethod
    @torch.autocast("cuda", dtype=torch.float32)
    def procrustes_alignment(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
        M = (pc1.mT @ pc2).mT.to(dtype=torch.float32)

        # Promote to at least float32 for svd
        u, _, v = torch.linalg.svd(M, full_matrices=False)
        d = u.det() * v.det()
        if d.ndim == 0:
            s = torch.tensor([1, 1, d.item()]).diag()
            assert s.shape == (3, 3)
        else:
            s = torch.stack([torch.tensor([1, 1, ds]).diag() for ds in d])
            assert s.shape == (pc1.shape[0], 3, 3)

        s = s.to(u)
        R = u.matmul(s).matmul(v).mT.to(pc1)

        return R
