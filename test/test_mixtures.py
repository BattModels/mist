from electrolyte_fm.models.polynomial_task_head import (
    BezierFourthPredictionTaskHead,
    LegendrePredictionTaskHead,
)
import torch


def test_legendre():
    task_head = LegendrePredictionTaskHead(
        embed_dim=1, polynomial_order=4, n_components=2
    )
    x = torch.ones(2)
    assert (task_head.legendre_poly(n=0, x=x) == torch.ones(1)).all()
    assert (task_head.legendre_poly(n=1, x=x) == x).all()
    assert (
        task_head.legendre_poly(n=8, x=x)
        == 0.125 * (63 * torch.pow(x, 5) - 70 * torch.pow(x, 3) + 15 * torch.pow(x, 1))
    ).all()


def test_zeros():
    m = BezierFourthPredictionTaskHead(32)
    assert torch.isclose(m.parametric_var[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(m.parametric_var[-1], torch.tensor(1.0), atol=1e-6)


def test_basis():
    m = BezierFourthPredictionTaskHead(32, polynomial_order=4)
    b = m.compute_basis(m.parametric_var)
    assert torch.linalg.matrix_rank(b) == m.polynomial_order
