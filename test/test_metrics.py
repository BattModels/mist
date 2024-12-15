from itertools import chain, repeat
from typing import Dict

import pytest
import torch
from torchmetrics import MetricCollection, Accuracy
from torchmetrics.regression import MeanSquaredError
from torchmetrics.wrappers import BootStrapper

from electrolyte_fm.utils.metrics import (
    get_metric,
    masked_loss,
    masked_metric_update,
    bootstrap_collection,
    OOVMetric,
    HotellingTwoSample,
    TokenCounter,
    IGNORE_INDEX,
)

BINARY_METRICS = ["auroc", "avg-precision", "crosstab"]

REGRESSION_METRICS = ["mae", "rmse", "r2", "mape"]


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
    metric = OOVMetric(
        Accuracy(task="binary", ignore_index=IGNORE_INDEX), unk_token_id=1
    )
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


def test_hotellings():
    metric = HotellingTwoSample(num_outputs=4, unk_token_id=1)
    rng = torch.manual_seed(0)
    p = 8
    batches = 8
    batch_size = 32
    for i in range(batches):
        preds = torch.randn(batch_size, p, generator=rng)
        targets = torch.randn(batch_size, p, generator=rng)
        input_ids = torch.randint(10, (batch_size, 16), generator=rng)
        metric.update(preds, targets, input_ids)

    out = metric.compute()
    assert isinstance(out, dict)
    assert isinstance(out["t2"], torch.FloatTensor) and out["t2"].ndim == 0
    assert isinstance(out["t_fdist"], torch.FloatTensor) and out["t_fdist"].ndim == 0
    assert isinstance(out["df"], int) and 0 < out["df"]
    assert isinstance(out["p"], int) and out["p"] == p
    assert isinstance(out["d2"], int) and out["d2"] == batch_size * batches - p - 1


def test_binary_stats():
    metric = OOVMetric(get_metric("crosstab", "binary"), 1)
    preds = torch.randn(32, 8)
    targets = torch.randint(1, (32, 8))
    input_ids = torch.ones(32, 8)
    metric.update(preds, targets, input_ids)
    out = metric.compute()
    assert out["tp_oov"] + out["tn_oov"] + out["fp_oov"] + out["fn_oov"] == 32 * 8


def test_token_counter():
    metric = TokenCounter()
    attention_mask = torch.zeros((32, 8))
    attention_mask[0][1] = 1
    attention_mask[0][2] = 1
    attention_mask[1][3] = 1
    labels = torch.full((32, 8), -100)
    labels[0][1] = 97
    metric.update(attention_mask, labels)
    out = metric.compute()
    assert out["masked_tokens"] == 1
    assert out["total_tokens"] == 3


def test_bootstrap():
    og_metrics = MetricCollection(
        {
            "mae": get_metric("mae", "regression", 1),
            "r2": get_metric("r2", "regression", 8),
            "auroc": get_metric("auroc", "binary", 8),
        }
    )
    metrics = bootstrap_collection(og_metrics)
    metrics = metrics.clone(prefix="train/")

    for i in range(10):
        preds = torch.rand(32, 8)
        targets = torch.rand(32, 8)
        metrics.update(preds, targets)
    out = metrics.compute()

    for v in out.values():
        assert isinstance(v, torch.Tensor)

    # Check metrics
    for key in set(og_metrics.keys()):
        for v in ["mean", "std"]:
            assert "/" not in key
            assert "_" not in key
            assert f"train/{key}_{v}" in out.keys()


def test_bootstrap_oov():
    og_metrics = MetricCollection(
        {
            "mae": get_metric("mae", "regression", 1),
            "r2": get_metric("r2", "regression", 8),
            "auroc": get_metric("auroc", "binary", 8),
        }
    )
    metrics = bootstrap_collection(og_metrics)
    metrics = OOVMetric(metrics.clone(prefix="train/"), unk_token_id=1)

    for i in range(10):
        preds = torch.rand(32, 8)
        targets = torch.rand(32, 8)
        input_ids = torch.randint(0, 8, (32, 8))
        mask = input_ids == 0
        masked_metric_update(metrics, preds, targets, mask, input_ids)
    out = metrics.compute()

    for v in out.values():
        assert isinstance(v, torch.Tensor)

    # Check metrics
    for key in set(og_metrics.keys()):
        for v in ["mean", "std"]:
            assert "/" not in key
            assert "_" not in key
            for token_group in ["oov", "non_oov", "all"]:
                assert f"train/{key}_{v}_{token_group}" in out.keys()
