"""Regression label masks must preserve native metrics and valid target values."""

import pytest
import torch
from torchmetrics.classification import AUROC, AveragePrecision
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

from electrolyte_fm.utils.metrics import (
    IGNORE_INDEX,
    OOVMetric,
    bootstrap_collection,
    get_metric,
    get_metrics,
    masked_metric_update,
)


def native(name):
    return {
        "mae": MeanAbsoluteError,
        "rmse": lambda: MeanSquaredError(squared=False),
        "mape": MeanAbsolutePercentageError,
        "r2": R2Score,
    }[name]()


@pytest.mark.parametrize("name", ["mae", "rmse", "mape", "r2"])
def test_dense_regression_metrics_match_native(name):
    targets = torch.tensor([[1.0, 8.0], [2.0, 3.0], [6.0, 5.0], [4.0, 9.0]])
    preds = targets + torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, -2.0], [1.0, 3.0]])
    metric = get_metric(name, "regression")
    assert metric.higher_is_better == native(name).higher_is_better
    assert metric.is_differentiable == native(name).is_differentiable
    masked_metric_update(
        metric, preds[:2], targets[:2], torch.ones_like(targets[:2], dtype=bool)
    )
    masked_metric_update(
        metric, preds[2:], targets[2:], torch.ones_like(targets[2:], dtype=bool)
    )
    torch.testing.assert_close(metric.compute(), native(name)(preds, targets))
    direct = get_metric(name, "regression")
    torch.testing.assert_close(direct(preds, targets), native(name)(preds, targets))

    channels = get_metric(name + "-channel", "regression", target_channels=["a", "b"])
    masked_metric_update(channels, preds, targets, torch.ones_like(targets, dtype=bool))
    for index, key in enumerate(["a", "b"]):
        torch.testing.assert_close(
            channels.compute()[key], native(name)(preds[:, index], targets[:, index])
        )


@pytest.mark.parametrize("name", ["mae", "rmse", "mape", "r2"])
def test_regression_masks_are_independent_by_channel(name):
    targets = torch.tensor([[1.0, 8.0], [2.0, 3.0], [6.0, 5.0], [4.0, 9.0]])
    preds = targets + torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, -2.0], [1.0, 3.0]])
    mask = torch.tensor([[True, False], [True, True], [False, True], [True, True]])
    targets[~mask] = float("nan")
    preds[~mask] = float("inf")
    metrics = get_metrics(
        [name, name + "-channel"],
        "regression",
        num_outputs=2,
        target_channels=["a", "b"],
    )
    for rows in [slice(0, 2), slice(2, 4)]:
        masked_metric_update(metrics, preds[rows], targets[rows], mask[rows])
    result = metrics.compute()
    expected = []
    for index, key in enumerate(["a", "b"]):
        valid = mask[:, index]
        value = native(name)(preds[valid, index], targets[valid, index])
        expected.append(value)
        torch.testing.assert_close(result[f"{name}_channel_{key}"], value)
    overall = (
        torch.stack(expected).mean()
        if name == "r2"
        else native(name)(preds[mask], targets[mask])
    )
    torch.testing.assert_close(result[name], overall)


def test_minus_100_is_a_valid_regression_target():
    targets = torch.tensor([[-100.0, 1.0], [2.0, -100.0], [3.0, 4.0]])
    preds = torch.tensor([[-97.0, 2.0], [3.0, -96.0], [9.0, 10.0]])
    mask = torch.tensor([[True, True], [True, True], [False, False]])
    metrics = get_metrics(
        ["mae", "mae-channel"], "regression", num_outputs=2, target_channels=["a", "b"]
    )
    masked_metric_update(metrics, preds, targets, mask)
    result = metrics.compute()
    torch.testing.assert_close(result["mae"], native("mae")(preds[mask], targets[mask]))
    for index, key in enumerate(["a", "b"]):
        torch.testing.assert_close(
            result[f"mae_channel_{key}"],
            native("mae")(preds[:2, index], targets[:2, index]),
        )


@pytest.mark.parametrize("name", ["mae", "rmse", "mape", "r2"])
def test_all_missing_channel_stays_undefined_and_can_receive_later_labels(name):
    targets = torch.tensor([[1.0, 8.0], [2.0, 3.0], [6.0, 5.0]])
    preds = targets + 1
    mask = torch.tensor([[True, False], [True, False], [True, False]])
    metric = get_metric(name + "-channel", "regression", target_channels=["a", "b"])
    masked_metric_update(metric, preds, targets, mask)
    result = metric.compute()
    assert torch.isnan(result["b"])
    torch.testing.assert_close(result["a"], native(name)(preds[:, 0], targets[:, 0]))
    masked_metric_update(metric, preds, targets, ~mask)
    for index, key in enumerate(["a", "b"]):
        torch.testing.assert_close(
            metric.compute()[key], native(name)(preds[:, index], targets[:, index])
        )


@pytest.mark.parametrize("bad_prediction", [float("nan"), float("inf")])
def test_valid_label_with_nonfinite_prediction_is_not_discarded(bad_prediction):
    metric = get_metric("mae-channel", "regression", target_channels=["a", "b"])
    masked_metric_update(
        metric,
        torch.tensor([[bad_prediction, 3.0]]),
        torch.tensor([[2.0, 2.0]]),
        torch.ones((1, 2), dtype=bool),
    )
    assert not torch.isfinite(metric.compute()["a"])
    assert torch.isfinite(metric.compute()["b"])


@pytest.mark.parametrize(
    "name,constructor", [("auroc", AUROC), ("avg-precision", AveragePrecision)]
)
def test_classification_ignore_index_behavior_is_unchanged(name, constructor):
    preds = torch.tensor([[0.1, 0.8], [0.4, 0.9], [0.6, 0.3]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True], [True, False]])
    metric = get_metric(name, "binary")
    masked_metric_update(metric, preds, targets, mask, int_cast=True)
    expected = constructor(
        task="binary",
        ignore_index=IGNORE_INDEX,
        thresholds=500 if name == "auroc" else 250,
    )
    torch.testing.assert_close(
        metric.compute(),
        expected(preds, targets.masked_fill(~mask, IGNORE_INDEX).int()),
    )


def test_oov_and_bootstrap_wrappers_keep_regression_masks():
    targets = torch.tensor([[-100.0], [2.0], [4.0], [8.0]])
    preds = torch.tensor([[-99.0], [3.0], [float("nan")], [9.0]])
    mask = torch.tensor([[True], [True], [False], [True]])
    ids = torch.tensor([[1], [0], [1], [0]])
    metrics = OOVMetric(get_metrics(["mae"], "regression"), unk_token_id=1)
    masked_metric_update(metrics, preds, targets, mask, ids)
    assert all(value == 1 for value in metrics.compute().values())
    bootstrapped = bootstrap_collection(
        get_metrics(["mae"], "regression"),
        num_bootstraps=5,
        sampling_strategy="multinomial",
    )
    masked_metric_update(bootstrapped, preds, targets, mask)
    assert bootstrapped.compute()["mae_mean"] == 1


@pytest.mark.parametrize("name", ["mae", "rmse", "mape", "r2"])
def test_empty_batches_do_not_change_later_results_or_checkpoint_state(name):
    targets = torch.tensor([[1.0, 8.0], [2.0, 3.0], [6.0, 5.0]])
    preds = targets + 1
    metric = get_metric(name, "regression")
    masked_metric_update(metric, preds, targets, torch.zeros_like(targets, dtype=bool))
    masked_metric_update(metric, preds, targets, torch.ones_like(targets, dtype=bool))
    torch.testing.assert_close(metric.compute(), native(name)(preds, targets))
    assert not metric.state_dict()
    metric.reset()
    masked_metric_update(metric, preds, targets, torch.ones_like(targets, dtype=bool))
    torch.testing.assert_close(metric.compute(), native(name)(preds, targets))


def test_pearson_channel_can_receive_labels_after_empty_batches():
    targets = torch.tensor([[1.0, 8.0], [2.0, 3.0], [6.0, 5.0], [4.0, 9.0]])
    preds = targets + torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, -2.0], [1.0, 3.0]])
    mask = torch.tensor([[True, False], [True, True], [False, True], [True, True]])
    metric = get_metric("pearson-channel", "regression", target_channels=["a", "b"])
    masked_metric_update(metric, preds, targets, torch.zeros_like(mask))
    masked_metric_update(metric, preds, targets, mask)
    for index, key in enumerate(["a", "b"]):
        valid = mask[:, index]
        torch.testing.assert_close(
            metric.compute()[key],
            PearsonCorrCoef()(preds[valid, index], targets[valid, index]),
        )
