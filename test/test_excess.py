from unittest.mock import Mock, patch
from pathlib import Path
import pytest
import torch
from torch import nn
import random

from transformers import AutoConfig
from electrolyte_fm.models.polynomials import LagrangePolynomial
from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsModel,
    ExcessPhysicsConfig,
    pairwise_fusion,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 41
    random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.parametrize(
    "name,n_targets",
    [
        (name, n_targets)
        for name in ["square-difference", "gaussian", "difference", "softmax"]
        for n_targets in [1, 2, 3]
    ],
)
def test_fusion(name, n_targets):
    model = pairwise_fusion(name, 6, 4, n_targets=n_targets)
    a = torch.rand(8, 6)
    b = torch.rand(8, 6)
    model = model.eval()
    assert torch.allclose(model(a, b), model(b, a).flip(-1))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_on_node(order):
    interp = LagrangePolynomial(polynomial_order=order)
    idx = random.randint(0, interp.polynomial_order - 1)
    x = interp.nodes[idx]
    coefs = torch.rand(order)
    y = interp(coefs, x)
    assert y.item() == coefs[idx]


@pytest.mark.parametrize("order", [2, 3, 4])
def test_eval_matches_linear(order):
    interp = LagrangePolynomial(polynomial_order=order)
    f = lambda x: 2 * x + 1  # noqa: E731
    coefs = torch.tensor([f(x) for x in interp.nodes])
    x = torch.rand(1)
    y = interp(coefs, x)
    assert torch.allclose(y, f(x), atol=1e-6)


@pytest.mark.parametrize("order", [2, 3, 4])
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


@pytest.mark.parametrize("order", [3, 4])
def test_on_node_gradient(order):
    interp = LagrangePolynomial(polynomial_order=order, zero_endpoints=True)
    x = interp.active_nodes.clone()
    N = len(interp.active_nodes)

    # Construct coefficients
    coefs = torch.rand((N))
    coefs = torch.stack([coefs.clone() for _ in range(N)])
    coefs.requires_grad = True

    # Check value
    y = interp(coefs, x)
    assert torch.allclose(y, coefs[0])

    # Check gradients
    y.sum().backward()
    assert coefs.grad is not None
    assert coefs.grad.isnan().count_nonzero() == 0
    assert coefs.grad.isinf().count_nonzero() == 0


class MockedOutput:
    def __init__(self, x) -> None:
        self.last_hidden_state = x


class MockedEncoder(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.mlp = nn.Linear(1, E)

    def forward(self, x, *args, **kwargs):
        return MockedOutput(self.mlp(x.unsqueeze(-1)))


@pytest.mark.parametrize(
    "interaction", ["square-difference", "gaussian", "difference", "softmax"]
)
def test_excess(interaction):
    B = 5  # Batch size
    C = 3  # Number of components
    L = 10  # Sequence length
    E = 8  # Embedding size
    P = 3  # polynomial_order
    T = 2  # Number of targets

    with patch("transformers.AutoModel.from_config") as mock_from_config:
        mock_from_config.side_effect = lambda config: MockedEncoder(config.hidden_size)
        encoder_config = Mock()
        encoder_config.hidden_size = E
        model = ExcessPhysicsModel(
            ExcessPhysicsConfig(
                encoder=encoder_config,
                target_columns=range(T),
                interactions=interaction,
                num_control=P,
            )
        )
    model = model.eval()

    input_ids = torch.rand(B, C, L)
    attention_mask = torch.rand(B, C, L) > 0.2
    composition = rand_simplex(B, C)
    temperature = torch.rand(B)

    # Evaluate model
    y, y_linear, y_excess = model(
        input_ids,
        attention_mask,
        composition=composition,
        temperature=temperature,
    )
    assert y.shape == (B, T)
    assert y_linear.shape == (B, T)
    assert y_excess.shape == (B, T)
    print("y:", y)

    # Check repeatable in eval
    y2 = model(
        input_ids,
        attention_mask,
        composition=composition,
        temperature=temperature,
    )[0]
    print("y2:", y2)
    assert torch.allclose(y, y2)

    # Check that the model is differentiable
    y.sum().backward()
    assert next(model.parameters()) is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} are missing"

    # Check that the predictions are permutation invariant
    for _ in range(C):
        pdx = torch.randperm(C)
        print("pdx:", pdx)
        y_perm = model(
            input_ids[:, pdx, :],
            attention_mask[:, pdx, :],
            composition=composition[:, pdx],
            temperature=temperature,
        )[0]
        print("y_perm:", y_perm)
        assert torch.allclose(y, y_perm)


def rand_simplex(*dims) -> torch.Tensor:
    x = torch.rand(*dims)
    x /= x.sum(dim=-1, keepdim=True)
    return x.detach()


@pytest.mark.parametrize(
    "config",
    [
        ExcessPhysicsModel.from_pretrained_encoder(
            "ibm/MoLFormer-XL-both-10pct"
        ).config,
        ExcessPhysicsConfig(encoder=AutoConfig.for_model("roberta")),
    ],
)
def test_config(config):
    d = config.to_dict()
    assert isinstance(d, dict)
    c2 = ExcessPhysicsConfig.from_dict(d)
    assert isinstance(c2, ExcessPhysicsConfig)
    assert config.to_dict() == c2.to_dict()


def test_saving(tmp_path):
    model = ExcessPhysicsModel(
        ExcessPhysicsConfig(
            encoder=AutoConfig.for_model(
                "roberta",
                hidden_size=512,
                intermediate_size=512,
                num_attention_heads=8,
                num_layers=2,
            )
        )
    )
    save_directory = Path(tmp_path, "model")
    model.save_pretrained(save_directory)
    assert save_directory.joinpath("config.json").exists()
    assert save_directory.joinpath("model.safetensors").exists()
    model2 = ExcessPhysicsModel.from_pretrained(save_directory)
    assert model2.config.to_dict() == model.config.to_dict()
