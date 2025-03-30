import logging
from typing import Union
from itertools import chain, repeat
from math import cos, sin

import pytest
import torch
from torchmetrics import Metric, Accuracy
from torchmetrics import MetricCollection as TmMetricCollection
from torchmetrics.wrappers import BootStrapper, ClasswiseWrapper
from scipy.linalg import orthogonal_procrustes

from electrolyte_fm.utils import metrics
from electrolyte_fm.utils.metrics import (
    MetricCollection,
    get_metric,
    get_metrics,
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

CHANNEL_METRICS = [
    ("auroc-channel", "binary"),
    ("mae-channel", "regression"),
    ("mape-channel", "regression"),
    ("r2-channel", "regression"),
]


@pytest.fixture(
    params=chain(
        zip(BINARY_METRICS, repeat("binary")),
        zip(REGRESSION_METRICS, repeat("regression")),
    ),
    ids=lambda x: x[0],
)
def metric(request):
    name, task_type = request.param
    return get_metric(name, task_type)


def is_binary(metric: Metric) -> bool:
    if "classification" in metric.__module__:
        return True
    elif isinstance(metric, ClasswiseWrapper):
        return is_binary(metric.metric)
    elif isinstance(metric, MetricCollection):
        return any(is_binary(m) for m in metric.values())
    return False


def test_is_binary():
    assert is_binary(get_metric("auroc", "binary"))
    assert is_binary(get_metric("crosstab", "binary"))
    assert not is_binary(get_metric("mae", "regression"))


def test_scalar(metric: Metric):
    assert hasattr(metric, "name") and isinstance(metric.name, str)
    B = 8
    C = 2
    preds = torch.rand(B, C)
    if is_binary(metric):
        targets = torch.randint(0, 1, (B, C))
    else:
        targets = torch.rand(B, C)

    out = metric(preds, targets)
    if isinstance(out, dict):
        for v in out.values():
            assert v.ndim == 0
    else:
        assert out.ndim == 0


def test_higher_is_better(metric):
    assert hasattr(metric, "higher_is_better")
    assert isinstance(metric.higher_is_better, Union[bool, None])


def test_invalid():
    with pytest.raises(ValueError):
        get_metric("auroc", "regression")
    with pytest.raises(ValueError):
        get_metric("not-a-metric", "multiclass")


@pytest.mark.parametrize("name", BINARY_METRICS)
def test_masked_metric(name):
    metric = get_metric(name, "binary")
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    input_ids = torch.randint(0, 3, (2, 3))
    mask = torch.tensor([[True, True, True], [False, True, True]])
    masked_metric_update(metric, preds, targets, mask, input_ids)
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
    masked_metric_update(metric, preds, targets, mask, input_ids)
    out_targets = metric.compute()
    assert out_targets == out_init

    # Repeat, changing the masked prediction
    metric.reset()
    preds[1, 0] = 0.3
    masked_metric_update(metric, preds, targets, mask, input_ids)
    assert out_init == metric.compute()


def test_metric_collection(metric):
    mc = MetricCollection({metric.name: metric})

    for _ in range(10):
        preds = torch.rand(8, 2)
        if is_binary(metric):
            targets = torch.randint(1, (8, 2))
        else:
            targets = torch.rand(8, 2)
        mc.update(preds, targets)

    out = mc.compute()
    assert isinstance(out, dict)
    if metric.name == "crosstab":
        keys = ["tp", "tn", "fp", "fn", "sup"]
        keys = ["crosstab_" + k for k in keys]
    else:
        keys = set([metric.name])

    assert set(keys) == set(out.keys())


def test_safe_for_nullset(metric):
    metric.compute()


def test_masked_loss():
    lossfn = torch.nn.MSELoss(reduction="none")
    preds = torch.rand(2, 3)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[True, True, False], [False, True, True]])

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
    mask = torch.tensor([[True, True, False], [False, True, True]])
    with pytest.raises(RuntimeError, match="Reduction must be 'none'"):
        masked_loss(lossfn, preds, targets, mask)


def test_auroc():
    auroc = get_metric("auroc", "binary")
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


def test_oov_metric_empty_group(metric):
    metric_oov = OOVMetric(metric, unk_token_id=1)
    preds = torch.rand(8, 3)
    targets = torch.randint(1, (8, 3))

    # Check for all non_oov
    input_ids = torch.zeros((8, 8))
    out = metric_oov(preds, targets, input_ids)
    assert isinstance(out, dict)

    # Check that all keys are present
    base_out = metric(preds, targets)
    if isinstance(base_out, dict):
        for k in base_out.keys():
            assert f"{k}_oov" in out and f"{k}_non_oov" in out and f"{k}_all" in out
    else:
        assert "oov" in out and "non_oov" in out and "all" in out

    # Check for all oov
    input_ids = torch.ones((8, 8))
    out = metric_oov(preds, targets, input_ids)
    if isinstance(base_out, dict):
        for k in base_out.keys():
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


@pytest.fixture(
    params=[
        {
            "metrics": ["mae", "r2"],
            "task": "regression",
        },
        {
            "metrics": ["auroc", "crosstab"],
            "task": "binary",
        },
        {
            "metrics": ["mae", "mae-channel", "r2", "r2-channel"],
            "task": "regression",
            "num_outputs": 3,
        },
        {
            "metrics": ["auroc", "auroc-channel", "avg-precision-channel"],
            "task": "binary",
            "num_outputs": 3,
        },
        {
            "metrics": ["auroc", "auroc-channel"],
            "task": "binary",
            "num_outputs": 1,
        },
    ],
)
def metric_collection(request):
    mc = get_metrics(**request.param)
    mc.__num_outputs__ = request.param.get("num_outputs", None)
    return mc


def test_bootstrap(metric_collection):
    og_metrics = metric_collection
    metrics = bootstrap_collection(og_metrics)
    metrics = metrics.clone(prefix="train/")
    og_metrics = og_metrics.clone(prefix="train/")
    print("metrics:", metrics)
    print("og_metrics:", og_metrics)

    C = og_metrics.__num_outputs__ or 4
    for i in range(10):
        preds = torch.rand(32, C)
        if is_binary(og_metrics):
            targets = torch.randint(0, 1, (32, C))
        else:
            targets = torch.rand(32, C)
        metrics.update(preds, targets)
        og_metrics.update(preds, targets)
    og_out = og_metrics.compute()
    print("og_out:", og_out)
    out = metrics.compute()
    print("out:", out)

    for v in out.values():
        assert isinstance(v, torch.Tensor)

    # Check metrics
    for key in og_out.keys():
        if "crosstab" in key:
            assert key in out.keys()
        else:
            for v in ["mean", "std"]:
                assert f"{key}_{v}" in out.keys()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/torchmetrics/issues/2046")
def test_torchmetricmetric_collection_issue():
    """Test case to validate torchmetrics.MetricCollection doesn't handle dict returns

    Related: https://github.com/Lightning-AI/torchmetrics/issues/2046

    If this test passes (unexpected), then should revert to torchmetrics.MetricCollection

    """
    mc = TmMetricCollection({"mae": BootStrapper(get_metric("mae", "regression"))})
    preds = torch.rand(8, 3)
    targets = torch.rand(8, 3)
    out = mc(preds, targets)
    assert out.keys() == set(["mae_mean", "mae_std"])


def test_our_metric_collection_issue():
    mc = MetricCollection({"mae": BootStrapper(get_metric("mae", "regression"))})
    preds = torch.rand(8, 3)
    targets = torch.rand(8, 3)
    out = mc(preds, targets)
    assert out.keys() == set(["mae_mean", "mae_std"])


def test_bootstrap_oov():
    og_metrics = MetricCollection(
        {
            "mae": get_metric("mae", "regression"),
            "r2": get_metric("r2", "regression"),
            "auroc": get_metric("auroc", "binary"),
        }
    )
    metrics = bootstrap_collection(og_metrics)
    metrics = OOVMetric(metrics.clone(prefix="train/"), unk_token_id=1)

    for i in range(10):
        preds = torch.rand(32, 8)
        targets = torch.rand(32, 8)
        input_ids = torch.randint(0, 8, (32, 8))
        mask = input_ids != 0
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


def test_ortho_procrustes():
    # Shifts shouldn't increase the loss
    B = 1
    N = 16
    atol = 1e-4
    x = torch.rand(B, N, 3)
    y = x + torch.rand(1, 3)

    def check_alignment(x, y, matches=True):
        R_ref, _ = orthogonal_procrustes(x.numpy()[0], y.numpy()[0])
        R = metrics.OrthoProcrustes.procrustes_alignment(x, y)
        loss = metrics.OrthoProcrustes.procrustes_disparity(x[0], y[0])
        logging.debug({"R": R, "R_ref": R_ref, "loss": loss, "x": x, "y": y})
        # assert R.isclose(torch.tensor(R_ref.tolist()), atol=1e-5).all()
        assert loss.isclose(torch.tensor(0.0), atol=atol) == matches

    check_alignment(x, y)

    # Or rotations
    def rot(θ=1):
        return torch.tensor([[cos(θ), -sin(θ), 0], [sin(θ), cos(θ), 0], [0, 0, 1]])

    y = x @ rot().reshape(1, 3, 3)
    check_alignment(x, y)

    # Or random orthogonal transforms
    y = x @ torch.nn.init.orthogonal_(torch.ones(3, 3))
    check_alignment(x, y)

    # Skewing the input does
    y = x @ torch.rand(3, 3)
    check_alignment(x, y, matches=False)
