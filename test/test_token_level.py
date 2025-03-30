import torch

from electrolyte_fm.models import token_level


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
