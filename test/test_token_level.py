import pytest
import torch

from electrolyte_fm.models import token_level
from electrolyte_fm.models.prediction_task_head import BiPairwiseBlock


def test_pairwise_distance_loss():
    B = 4
    S = 8
    pairwise = torch.rand(B, S, S).pow(2)
    coords = torch.rand(B, S, 3)
    mask = torch.rand(B, S) > 0.5
    loss = token_level.pairwise_distance_loss(pairwise, coords, mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0

    # Loss should be zero if all pairs are the same
    pairwise = torch.cdist(coords, coords)
    loss = token_level.pairwise_distance_loss(pairwise, coords, mask)
    assert loss.isclose(torch.tensor(0.0))


@pytest.mark.parametrize("B, N, D", [(2, 4, 8), (1, 2, 16), (3, 5, 6)])
def test_output_shape(B, N, D):
    """Test whether the output has the correct shape (B, N, N, D)."""
    model = BiPairwiseBlock(D)
    x = torch.randn(B, N, D)
    y = model(x)
    assert y.shape == (B, N, N, D), f"Expected shape {(B, N, N, D)}, got {y.shape}"


def test_gradients():
    """Ensure that gradients properly flow through all parameters."""
    B, N, D = 2, 4, 8
    model = BiPairwiseBlock(D)
    x = torch.randn(B, N, D, requires_grad=True)
    y = model(x).sum()  # Scalar loss
    y.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not found for {name}"

    # Check symmetric gradients for bi_weight
    bi_weight_grad = model.bi_weight.grad
    assert torch.norm(bi_weight_grad - bi_weight_grad.T).item() < 1e-6


def test_bias_effect():
    """Test that bias is applied correctly by comparing with zero input."""
    B, N, D = 2, 4, 8
    model = BiPairwiseBlock(D)
    x_zero = torch.zeros(B, N, D)
    y_zero = model(x_zero)

    # Bias should be added to every output position
    expected = model.bias.reshape(1, 1, 1, D).expand(B, N, N, D)
    assert torch.allclose(y_zero, expected, atol=1e-5), "Bias not applied correctly"


def test_symmetry():
    """BiPairwiseBlock should maintain symmetry"""
    B, N, D = 1, 3, 5
    model = BiPairwiseBlock(D)
    x = torch.randn(B, N, D)
    y = model(x)

    # Bilinear transform may not be symmetric, but should be structurally consistent
    y_t = y.transpose(1, 2)
    assert y.isclose(y_t).all(), "BiPairwiseBlock should be symmetric"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    """Test if the model runs on GPU and produces the same result as CPU."""
    B, N, D = 2, 4, 8
    dtype = torch.float64
    model = BiPairwiseBlock(D)
    model = model.to("cuda", dtype=dtype)
    x_cpu = torch.randn(B, N, D, dtype=dtype)

    x_gpu = x_cpu.to("cuda")
    y_gpu = model(x_gpu).cpu()
    y_cpu = model.cpu()(x_cpu)

    print(y_gpu)
    print(y_cpu - y_gpu)
    assert torch.allclose(
        y_cpu, y_gpu, atol=1e-5
    ), "Mismatch between CPU and GPU outputs"
