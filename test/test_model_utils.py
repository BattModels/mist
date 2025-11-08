import torch
from electrolyte_fm.models.model_utils import masked_mean_pool


def test_masked_mean_pool_basic():
    # Input tensor: shape (1, 4, 2)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

    # Mask: keep only the first 2 entries
    mask = torch.tensor([[1, 1, 0, 0]])

    # Expected mean over the first two vectors
    expected = torch.tensor([[2.0, 3.0]])

    result = masked_mean_pool(x, mask)
    assert torch.allclose(result, expected, atol=1e-5)
