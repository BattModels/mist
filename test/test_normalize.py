import torch
import logging
from torch.masked import MaskedTensor
import pytest
from electrolyte_fm.models.normalize import AbstractNormalizer
from datasets import IterableDataset

N_TARGETS = 4


@pytest.fixture(
    scope="session",
    params=["standardize", "power_transform", "log_transform", "identity"],
)
def fitted_normalizer(request):
    tf = AbstractNormalizer.get(request.param, N_TARGETS)
    ds = IterableDataset.from_generator(
        lambda: (
            {"target": torch.rand(N_TARGETS), "target_mask": torch.ones(N_TARGETS) == 1}
            for _ in range(100)
        )
    )
    tf.fit(ds)
    return tf


def test_standarize_fit():
    n = 4
    mu = 10 * torch.rand(n)
    std = 0.5 * torch.rand(n)

    def gen():
        for _ in range(1000):
            yield {
                "target": std * torch.randn(n) + mu,
                "target_mask": torch.rand(n) > 0.2,
            }

    ds = IterableDataset.from_generator(gen)
    tf = AbstractNormalizer.get("standardize", n)
    state = tf.fit(ds)
    logging.info({"state": state, "mu": mu, "std": std})
    assert "mean" in state and "std" in state
    assert state["mean"].isclose(mu, atol=1e-1).all()
    assert state["std"].isclose(std, atol=1e-1).all()


def test_serialization(fitted_normalizer: AbstractNormalizer):
    config = fitted_normalizer.to_config()
    r = AbstractNormalizer.get(config["class"], config["num_outputs"])
    assert r.num_outputs == fitted_normalizer.num_outputs
    assert r.to_config() == config
    assert r.__class__ == fitted_normalizer.__class__


def test_normalization(fitted_normalizer: AbstractNormalizer):
    x = torch.rand(5, N_TARGETS)
    y = fitted_normalizer.inverse(x)
    assert y.isfinite().all() and not y.isnan().any()
    z = fitted_normalizer.forward(y)
    assert z.isfinite().all() and not z.isnan().any()
    assert x.allclose(z, atol=1e-4)


def test_channel_wise():
    with pytest.raises(AssertionError):
        AbstractNormalizer.get(["standardize", "power_transform"], 1)

    tf = AbstractNormalizer.get(["standardize", "power_transform"], 2)
    x = MaskedTensor(torch.rand(100, 2), torch.ones(100, 2) == 1)
    init_state = tf.state_dict()
    tf._fit(x)
    assert tf.state_dict() != init_state

    x = torch.rand(5, 2)
    y = tf.inverse(x)
    assert y.isfinite().all() and not y.isnan().any()
    assert y[:, [0]].allclose(tf.transforms[0].inverse(x[:, [0]]))
    assert y[:, [1]].allclose(tf.transforms[1].inverse(x[:, [1]]))
    z = tf.forward(y)
    assert z.isfinite().all() and not z.isnan().any()
    assert x.allclose(z, atol=1e-4)
