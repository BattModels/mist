from itertools import chain, repeat
from typing import Dict

import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics import Accuracy

from electrolyte_fm.utils.metrics import (
    get_metric,
    masked_loss,
    masked_metric_forward,
    OOVMetric,
)

BINARY_METRICS = ["auroc", "avg-precision"]

REGRESSION_METRICS = ["mae", "avg-rmse", "r2"]


@pytest.mark.parametrize(
    "name,task_type",
    chain(
        zip(BINARY_METRICS, repeat("binary")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
)
def test_scalar(name, task_type):
    B = 8
    C = 4
    metric = get_metric(name, task_type, C)
    preds = torch.rand(B, C)
    if task_type == "binary":
        targets = torch.randint(0, 1, (B, C))
    else:
        targets = torch.rand(B, C)

    out = metric(preds, targets)
    assert out.ndim == 0


def test_invalid():
    with pytest.raises(ValueError):
        get_metric("auroc", "regression", 1)
    with pytest.raises(ValueError):
        get_metric("not-a-metric", "multiclass", 8)


@pytest.mark.parametrize("name", BINARY_METRICS)
def test_masked_metrics(name):
    metric = get_metric(name, "binary", 3)
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[False, False, True], [True, False, False]])
    out_init = masked_metric_forward({name: metric}, preds, targets, mask)
    assert isinstance(out_init, Dict)
    assert name in out_init

    # Repeat, changing the masked targets
    metric.reset()
    targets[1, 0] = 1
    assert mask[1, 0]
    out_targets = masked_metric_forward({name: metric}, preds, targets, mask)
    assert out_init == out_targets

    # Repeat, changing the masked prediction
    metric.reset()
    preds[1, 0] = 0.3
    out_preds = masked_metric_forward({name: metric}, preds, targets, mask)
    assert out_init == out_preds


def test_masked_loss():
    lossfn = torch.nn.MSELoss()
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[False, False, True], [True, False, False]])

    loss = masked_loss(lossfn, preds, targets, mask)
    assert ~(loss.isnan()) and loss.isfinite() and 0 < loss

    # Repeat, changing a masked target
    targets[1, 0] = 423.0
    assert mask[1, 0]
    loss_targets = masked_loss(lossfn, preds, targets, mask)
    assert loss_targets == loss

    # Repeat, changing a masked prediction
    targets[0, 3] = 616.0
    assert mask[0, 3]
    loss_preds = masked_loss(lossfn, preds, targets, mask)
    assert loss_preds == loss


def test_oov_metric():
    metric = OOVMetric(Accuracy(task="binary"), unk_token_id=1)
    out = metric(
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([[3, 1], [0, 0], [3, 2]]),
    )
    assert isinstance(out, dict) and "oov" in out and "non_oov" in out and "all" in out
    assert out["oov"] == torch.tensor(0)
    assert out["non_oov"] == torch.tensor(0.5)
    assert out["all"] == torch.tensor(1 / 3)


def test_oov_metric_collection():
    mc = MetricCollection(
        {"foo": Accuracy(task="binary"), "bar": Accuracy(task="binary")}
    )
    metric = OOVMetric(mc, unk_token_id=1)
    out = metric(
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([[3, 1], [0, 0], [3, 2]]),
    )
    assert isinstance(out, dict)
    for k in mc.keys():
        for oov_key in ["oov", "non_oov", "all"]:
            assert f"{k}_{oov_key}" in out

        assert out[f"{k}_oov"] == torch.tensor(0)
        assert out[f"{k}_non_oov"] == torch.tensor(0.5)
        assert out[f"{k}_all"] == torch.tensor(1 / 3)
