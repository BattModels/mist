import torch
import pytest
from src.generate import QuadrantCritic


def test_all_values_within_bounds():
    limits = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    channels = ["a", "b"]
    critic = QuadrantCritic(limits, channels)

    y = torch.tensor([[0.5, 0.0], [0.1, -0.5]])
    out = critic(y).all(dim=1)
    assert torch.all(out == torch.tensor([True, True]))


def test_values_outside_bounds():
    limits = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    channels = ["a", "b"]
    critic = QuadrantCritic(limits, channels)

    y = torch.tensor([[1.5, 0.0], [0.5, -1.5]])
    out = critic(y).all(dim=1)
    assert torch.all(out == torch.tensor([False, False]))


def test_mixed_validity_rows():
    limits = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    channels = ["a", "b"]
    critic = QuadrantCritic(limits, channels)

    y = torch.tensor([[0.5, 0.0], [1.5, 0.0], [0.5, 2.0]])
    out = critic(y).all(dim=1)
    assert torch.all(out == torch.tensor([True, False, False]))


def test_unconstrained_channels():
    limits = {"a": (0.0, 1.0)}  # b is unconstrained
    channels = ["a", "b"]
    critic = QuadrantCritic(limits, channels)

    y = torch.tensor([[0.5, 999.0], [0.1, -999.0]])
    out = critic(y).all(dim=1)
    assert torch.all(out == torch.tensor([True, True]))


def test_values_equal_to_bounds_are_false():
    limits = {"a": (0.0, 1.0)}
    channels = ["a"]
    critic = QuadrantCritic(limits, channels)

    y = torch.tensor([[0.0], [1.0], [0.5]])
    out = critic(y).squeeze(1)
    assert torch.all(out == torch.tensor([False, False, True]))


def test_active_channels_mask():
    limits = {"a": (0.0, 1.0)}
    channels = ["a", "b", "c"]
    critic = QuadrantCritic(limits, channels)

    expected = torch.tensor([True, False, False])
    assert (critic.active_channels == expected).all()


def test_limits_not_subset_of_channels_raises():
    limits = {"a": (0.0, 1.0), "x": (0.0, 2.0)}  # x not in channels
    channels = ["a", "b"]
    with pytest.raises(AssertionError):
        _ = QuadrantCritic(limits, channels)
