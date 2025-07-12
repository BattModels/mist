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
        x = x.unsqueeze(-1)  # (..., 1)
        x_diff = x - self.nodes  # (..., n)
        on_node = x_diff.abs() <= 1e-8

        if self.zero_endpoints:
            zeros = torch.zeros_like(coefs[..., :1])
            coefs = torch.cat([zeros, coefs, zeros], dim=-1)

        assert coefs.shape[-1] == self.polynomial_order
        w_over_diff = self.weights / x_diff  # (..., N)
        num = (w_over_diff * coefs).sum(dim=-1)  # (...)
        den = w_over_diff.sum(dim=-1)  # (...)
        y = num / den

        # Handle exact matches
        if on_node.any():
            matched_idx = on_node.float().argmax(dim=-1)  # (...)
            gather_idx = matched_idx.unsqueeze(-1)  # (..., 1)
            y = torch.where(
                on_node.any(dim=-1),
                coefs.gather(dim=-1, index=gather_idx).squeeze(-1),
                y,
            )
        return y
