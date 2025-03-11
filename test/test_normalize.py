import torch
from torch.masked import MaskedTensor
import pytest
from electrolyte_fm.models.normalize import AbstractNormalizer


@pytest.fixture(params=["standardize", "power_transform", "log_transform", "identity"])
def fitted_normalizer(request):
    tf = AbstractNormalizer.get(request.param, 1)
    x = MaskedTensor(torch.rand(100, 1), torch.ones(100, 1) == 1)
    tf._fit(x)
    return tf


def test_serialization(fitted_normalizer: AbstractNormalizer):
    config = fitted_normalizer.to_config()
    r = AbstractNormalizer.get(config["class"], config["num_outputs"])
    assert r.num_outputs == fitted_normalizer.num_outputs
    assert r.to_config() == config
    assert r.__class__ == fitted_normalizer.__class__


def test_normalization(fitted_normalizer: AbstractNormalizer):
    x = torch.rand(5, 1)
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
