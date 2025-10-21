import pytest
import random
import torch
from electrolyte_fm.models.polynomials import LagrangePolynomial


@pytest.mark.parametrize("order", range(2, 8))
def test_on_node(order):
    interp = LagrangePolynomial(polynomial_order=order)
    idx = random.randint(0, interp.polynomial_order - 1)
    coefs = torch.rand(order)
    for idx, x in enumerate(interp.nodes):
        y = interp(coefs, x)
        assert torch.allclose(y, coefs[idx])


@pytest.mark.parametrize("order", range(2, 8))
def test_eval_matches_linear(order):
    interp = LagrangePolynomial(polynomial_order=order)
    f = lambda x: 2 * x + 1  # noqa: E731
    coefs = torch.tensor([f(x) for x in interp.nodes])
    x = torch.linspace(0, 1, steps=10)
    coefs = coefs.view(1, -1).expand(x.shape[0], -1)
    y = interp(coefs, x)
    assert torch.allclose(y, f(x), atol=1e-6)


@pytest.mark.parametrize("order", range(2, 8))
def test_batched_coefs(order):
    interp = LagrangePolynomial(polynomial_order=order)
    f1 = lambda x: 2 * x + 1  # noqa: E731
    f2 = lambda x: -2 * x + 3  # noqa: E731
    coefs = torch.tensor(
        [
            [f1(x) for x in interp.active_nodes],
            [f2(x) for x in interp.active_nodes],
        ]
    )
    x = torch.rand(2)
    y = interp(coefs, x)
    expected = torch.tensor([f1(x[0]), f2(x[1])])
    assert torch.allclose(y, expected, atol=1e-6)


@pytest.mark.parametrize("order,zero", [(3, False), (3, True), (4, True)])
def test_gradient(order, zero):
    interp = LagrangePolynomial(polynomial_order=order, zero_endpoints=zero)
    B, S = 2, 3
    N = len(interp.active_nodes)
    coefs = torch.rand((B, S, N), requires_grad=True)
    x = torch.rand((B, S))
    y = interp(coefs, x).sum()
    y.backward()
    assert coefs.grad is not None


def test_gradient_dx():
    interp = LagrangePolynomial(polynomial_order=4)
    f = lambda x: 2 * x + 1  # noqa: E731
    x = torch.linspace(0, 1, steps=10)
    coefs = torch.tensor([f(x) for x in interp.nodes])
    coefs = coefs.view(1, -1).expand(x.shape[0], -1)
    x.requires_grad = True
    coefs.requires_grad = True
    y = interp(coefs, x)
    assert torch.allclose(y, f(x), atol=1e-6)

    # Check gradients
    y.sum().backward()
    assert coefs.grad is not None
    assert coefs.grad.isnan().count_nonzero() == 0
    assert coefs.grad.isinf().count_nonzero() == 0
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor(2.0))


@pytest.mark.parametrize("order", [3, 4])
def test_on_node_gradient(order):
    interp = LagrangePolynomial(polynomial_order=order, zero_endpoints=True)
    x = interp.nodes.clone()
    N = len(interp.active_nodes)
    x.requires_grad = True

    # Construct coefficients
    coefs = torch.rand((N))
    coefs = coefs.view(1, -1).expand(x.shape[0], -1).detach()
    coefs.requires_grad = True

    # Check value
    y = interp(coefs, x)
    assert torch.allclose(y[1:-1], coefs[0])

    # Check gradients
    y.sum().backward()
    assert coefs.grad is not None
    assert coefs.grad.isnan().count_nonzero() == 0
    assert coefs.grad.isinf().count_nonzero() == 0
    assert x.grad is not None
    assert x.grad.isnan().count_nonzero() == 0
    assert x.grad.isinf().count_nonzero() == 0
