from itertools import chain, repeat
from typing import Dict

import pytest
import torch

from electrolyte_fm.utils.metrics import (
    get_metric,
    masked_loss,
    masked_metric_forward,
)

BINARY_METRICS = ["auroc", "avg-precision"]

REGRESSION_METRICS = ["mae", "avg-rmse", "r2"]


@pytest.mark.parametrize(
    "name,task_type",
    chain(
        zip(BINARY_METRICS, repeat("binary_classification")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
)
def test_scalar(name, task_type):
    B = 8
    C = 4
    metric = get_metric(name, task_type, C)
    preds = torch.rand(B, C)
    if task_type == "binary_classification":
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
    metric = get_metric(name, "binary_classification", 3)
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
