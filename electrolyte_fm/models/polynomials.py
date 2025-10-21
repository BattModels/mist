import torch
from torch import nn
from math import pi, cos


class PolynomialPredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.coeffients = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, polynomial_order),
        )

    def forward(self, emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        coefs = self.coeffients(emb)
        return self.eval_poly(coefs, x)

    def eval_poly(self, coefs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial with coefficients `coefs` at `x`"""
        raise NotImplementedError


class LagrangePolynomial(nn.Module):
    def __init__(self, polynomial_order: int = 4, zero_endpoints: bool = False):
        super().__init__()
        self.polynomial_order = polynomial_order
        self.zero_endpoints = zero_endpoints
        assert self.polynomial_order >= (2 if not zero_endpoints else 3)
        nodes = torch.tensor(
            [
                cos((2 * k + 1) * pi / (2 * self.polynomial_order))
                for k in range(self.polynomial_order)
            ]
        )
        nodes -= nodes.min()
        nodes /= nodes.max()
        self.register_buffer("nodes", nodes, persistent=False)
        self.register_buffer(
            "weights", self.barycentric_weights(self.nodes), persistent=False
        )
        self.register_buffer(
            "weights_mask",
            ~torch.eye(self.polynomial_order, dtype=torch.bool),
            persistent=False,
        )

    @property
    def active_nodes(self):
        if self.zero_endpoints:
            return self.nodes[1:-1]
        return self.nodes

    @classmethod
    def barycentric_weights(cls, nodes: torch.Tensor) -> torch.Tensor:
        """Compute barycentric weights for Lagrange interpolation"""
        assert nodes.ndim == 1
        x_diff = nodes.unsqueeze(0) - nodes.unsqueeze(1)  # (n, n)
        x_diff.fill_diagonal_(1.0)  # avoid zero division on diagonal
        w = 1.0 / x_diff.prod(dim=1)
        return w

    def forward(self, coefs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.zero_endpoints:
            zeros = torch.zeros_like(coefs[..., :1])
            coefs = torch.cat([zeros, coefs, zeros], dim=-1)
        assert coefs.shape[-1] == self.polynomial_order

        x = x.unsqueeze(-1)  # (..., 1)
        nodes = self.nodes  # (n,)
        n = self.polynomial_order

        # Compute all (x - x_k) terms
        x_diff = nodes - x  # (..., n)

        # Mask diagonal in x - x_k products for each basis function
        x_diff_masked = x_diff.unsqueeze(-2).expand(*x.shape[:-1], n, n)  # (..., n, n)
        x_diff_masked = x_diff_masked.masked_select(self.weights_mask)  # (..., n*(n-1))
        x_diff_masked = x_diff_masked.view(*x.shape[:-1], n, n - 1)

        num = x_diff_masked.prod(dim=-1)
        basis = num * self.weights
        y = (basis * coefs).sum(dim=-1)  # (...)

        return y
