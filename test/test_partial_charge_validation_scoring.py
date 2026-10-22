"""Check recovery arithmetic with fixed predictions; no model is constructed."""

import pytest
import torch

from opt.benchmarks.scripts.partial_charge import score_batches


def test_saved_prediction_scoring_uses_each_channel_mask():
    targets = torch.tensor([[[-100.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
    predictions = targets + torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    mask = torch.tensor([[[True, False, True, True], [True, True, False, True]]])
    predictions[~mask] = float("nan")
    targets[~mask] = float("nan")

    # score_batches only needs a callable and the channel names. This adapter
    # returns precomputed values and has no weights or model implementation.
    def saved_predictions(*args):
        return None, predictions

    saved_predictions.token_targets = ["a", "b", "c", "d"]
    batch = {
        "input_ids": torch.zeros((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "token_target": targets,
        "token_target_mask": mask,
    }
    overall, channels, rows = score_batches(
        saved_predictions, [batch], torch.device("cpu")
    )
    error = (predictions.double() - targets.double())[mask]
    assert overall["mae"] == pytest.approx(float(error.abs().mean()))
    assert overall["rmse"] == pytest.approx(float(error.square().mean().sqrt()))
    assert rows == 1
    for index, name in enumerate(saved_predictions.token_targets):
        valid = mask[..., index]
        error = (predictions[..., index].double() - targets[..., index].double())[valid]
        assert channels[name]["count"] == int(valid.sum())
        assert channels[name]["mae"] == pytest.approx(float(error.abs().mean()))
        assert channels[name]["rmse"] == pytest.approx(
            float(error.square().mean().sqrt())
        )
