from itertools import chain, repeat
from typing import Dict

import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics import Accuracy

from electrolyte_fm.utils.metrics import (
    get_metric,
    masked_loss,
    masked_metric_update,
    OOVMetric,
    IGNORE_INDEX,
)

BINARY_METRICS = ["auroc", "avg-precision", "crosstab"]

REGRESSION_METRICS = ["mae", "rmse", "r2"]


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
    if name == "crosstab":
        for v in out.values():
            assert v.ndim == 0
    else:
        assert out.ndim == 0


@pytest.mark.parametrize(
    "name,task_type",
    chain(
        zip(BINARY_METRICS, repeat("binary")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
)
def test_higher_is_better(name, task_type):
    m = get_metric(name, task_type, 2)
    if name == "crosstab":
        assert not hasattr(m, "higher_is_better")
    else:
        assert hasattr(m, "higher_is_better")
        assert isinstance(m.higher_is_better, bool)


def test_invalid():
    with pytest.raises(ValueError):
        get_metric("auroc", "regression", 1)
    with pytest.raises(ValueError):
        get_metric("not-a-metric", "multiclass", 8)


@pytest.mark.parametrize("name", BINARY_METRICS)
def test_masked_metric(name):
    metric = get_metric(name, "binary", 3)
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[False, False, True], [True, False, False]])
    masked_metric_update(metric, preds, targets, mask)
    out_init = metric.compute()
    if name == "crosstab":
        assert isinstance(out_init, dict)
        assert out_init.keys() == set(["tp", "tn", "fp", "fn", "sup"])
    else:
        assert isinstance(out_init, torch.FloatTensor)

    # Repeat, changing the masked targets
    metric.reset()
    targets[1, 0] = 1
    assert mask[1, 0]
    masked_metric_update(metric, preds, targets, mask)
    out_targets = metric.compute()
    assert out_targets == out_init

    # Repeat, changing the masked prediction
    metric.reset()
    preds[1, 0] = 0.3
    masked_metric_update(metric, preds, targets, mask)
    assert out_init == metric.compute()


@pytest.mark.parametrize(
    "name,task_type",
    chain(
        zip(BINARY_METRICS, repeat("binary")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
)
def test_safe_for_nullset(name, task_type):
    metric = get_metric(name, task_type, 1)
    metric.compute()


def test_masked_loss():
    lossfn = torch.nn.MSELoss(reduction="none")
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
    targets[0, 2] = 616.0
    assert mask[0, 2]
    loss_preds = masked_loss(lossfn, preds, targets, mask)
    assert loss_preds == loss


def test_masked_loss_reduction():
    lossfn = torch.nn.MSELoss()
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[False, False, True], [True, False, False]])
    with pytest.raises(RuntimeError, match="Reduction must be 'none'"):
        masked_loss(lossfn, preds, targets, mask)


def test_auroc():
    auroc = get_metric("auroc", "binary", 1)
    test_logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert auroc(test_logits, torch.tensor([0, 0, 1, 1])) == 1.0
    assert auroc(torch.tensor([0.4, 0.3, 0.2, 0.1]), torch.tensor([0, 1, 1, 0])) == 0.5
    assert auroc(test_logits, torch.tensor([1, 1, 0, 0])) == 0.0
    assert auroc(test_logits, torch.tensor([1, 0, 1, 1])) == 2.0 / 3
    # no false positives
    assert auroc(test_logits, torch.tensor([1, 1, 1, 1])) == 0.0
    # no true positives
    assert auroc(test_logits, torch.tensor([0, 0, 0, 0])) == 0.0


def test_oov_metric():
    metric = OOVMetric(Accuracy(task="binary", ignore_index=IGNORE_INDEX), unk_token_id=1)
    out = metric(
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([[3, 1], [0, 0], [3, 2]]),
    )
    assert isinstance(out, dict) and "oov" in out and "non_oov" in out and "all" in out
    assert out["oov"] == torch.tensor(0)
    assert out["non_oov"] == torch.tensor(0.5)
    assert out["all"] == torch.tensor(1 / 3)

@pytest.mark.parametrize(
    "name,task_type",
    chain(
        zip(BINARY_METRICS, repeat("binary")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
)
def test_oov_metric_empty_group(name, task_type):
    base_metric = get_metric(name, task_type, 3)
    metric = OOVMetric(base_metric, unk_token_id=1)
    preds = torch.rand(8, 3)
    targets = torch.randint(1, (8, 3))

    # Check for all non_oov
    input_ids = torch.zeros((8, 8))
    out = metric(preds, targets, input_ids)
    assert isinstance(out, dict) 
    if hasattr(base_metric, "keys"):
        for k in base_metric:
            assert f"{k}_oov" in out and f"{k}_non_oov" in out and f"{k}_all" in out
    else:
        assert "oov" in out and "non_oov" in out and "all" in out 

    # Check for all oov
    input_ids = torch.ones((8, 8))
    out = metric(preds, targets, input_ids)
    assert isinstance(out, dict) 
    if hasattr(base_metric, "keys"):
        for k in base_metric:
            assert f"{k}_oov" in out and f"{k}_non_oov" in out and f"{k}_all" in out
    else:
        assert "oov" in out and "non_oov" in out and "all" in out 

def test_oov_metric_is_oov():
    metric = OOVMetric(Accuracy(task="binary"), unk_token_id=1)
    out = metric(
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([[3, 1], [0, 1], [3, 1]]),  # This will be ignored
        torch.tensor([True, False, False]),  # As is_oov is provided
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
    metric_keys = list(metric.keys())
    for k in mc.keys():
        for oov_key in ["oov", "non_oov", "all"]:
            assert f"{k}_{oov_key}" in out
            assert f"{k}_{oov_key}" in metric_keys

        assert out[f"{k}_oov"] == torch.tensor(0)
        assert out[f"{k}_non_oov"] == torch.tensor(0.5)
        assert out[f"{k}_all"] == torch.tensor(1 / 3)


def test_oov_metric_masked():
    metric = OOVMetric(
        Accuracy(task="binary", ignore_index=IGNORE_INDEX),
        unk_token_id=1,
    )
    preds = torch.tensor([1, 0, 0])
    targets = torch.tensor([0, 1, 0])
    input_ids = torch.tensor([[3, 1], [0, 0], [3, 2]])
    mask = torch.tensor([False, True, False])

    # Initial variant
    masked_metric_update(metric, preds, targets, mask, input_ids)
    out = metric.compute()
    assert out["oov"] == torch.tensor(0)
    assert out["non_oov"] == torch.tensor(1)
    assert out["all"] == torch.tensor(0.5)

    # Repeat, changing a masked target
    metric.reset()
    targets[1] = False
    assert mask[1]
    masked_metric_update(metric, preds, targets, mask, input_ids)
    out_targets = metric.compute()
    assert out_targets == out

    # Repeat, changing a masked preds
    metric.reset()
    preds[1] = True
    assert mask[1]
    masked_metric_update(metric, preds, targets, mask, input_ids)
    out_preds = metric.compute()
    assert out_preds == out
