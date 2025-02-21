from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from .prediction_task_head import PredictionTaskHead


class PolynomialPredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        assert n_components == 2, "Only binary mixtures are supported"
        self.n_components = n_components
        embed_dim += 1  # Temperature appended to embedding
        self.include_linear_mixing = include_linear_mixing

        # predict property for single substance in mixture (P_i)
        self.single_substance_property = PredictionTaskHead(embed_dim=embed_dim)

        # predict polyn coefficients
        self.coeffients = PredictionTaskHead(embed_dim=embed_dim)

    def forward(self, batch):
        raise NotImplementedError


class RKPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

    def forward(self, batch):
        P_m = 0

        if self.include_linear_mixing:
            # linear mixing term
            for i in range(self.n_components):
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
                P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

            concat_embedding = tuple(
                batch[f"embedding_{i}"] for i in range(self.n_components)
            )
            concat_embedding = torch.hstack(concat_embedding)
            RK_coeffients = self.coeffients(concat_embedding)

        # excess term
        for i in range(self.n_components):
            x_i = batch[f"composition_{i}"]
            for j in range(i + 1, self.n_components):
                x_j = batch[f"composition_{j}"]
                x_ix_j = torch.mul(x_i, x_j)
                difference = torch.abs((x_i - x_j))
                for k in range(self.polynomial_order):
                    RK_summation = torch.mul(
                        RK_coeffients[:, k], torch.pow(difference, k)
                    )
                    P_m += torch.mul(x_ix_j, RK_summation).view(-1, 1)
        return P_m


class LegendrePredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 8,
        n_components: int = 2,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
        )

    def legendre_poly(self, n, x):
        if n == 0:
            # base case: T_0 = 1
            return torch.ones_like(x)
        elif n == 1:
            # base case: T_1 = x
            return x
        else:
            # recurrence relation for legendre polynomials
            # Bonnet's formula: (n+1)*T_{n} = (2*n+1)*x*T_{n-1} - n*T_{n-2}
            return torch.div(
                torch.mul((2 * n + 1) * x, self.legendre_poly(n - 1, x))
                - n * self.legendre_poly(n - 2, x),
                n + 1,
            )

    def forward(self, batch):
        P_m = 0

        if self.include_linear_mixing:
            # linear mixing term
            for i in range(self.n_components):
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
                P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        concat_embedding = tuple(
            batch[f"embedding_{i}"] for i in range(self.n_components)
        )

        concat_embedding = torch.hstack(concat_embedding)
        coeffients = self.coeffients(concat_embedding)

        # binary excess term
        x_i = batch["composition_0"]
        x_j = batch["composition_1"]
        x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]

        x = torch.abs(1 - 2 * x_j)

        for m in range(self.polynomial_order):
            summation = torch.mul(coeffients[:, m], self.legendre_poly(m, x))
            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # [batch_size, 1]


class ChebyshevPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
        )

    def chebyshev_poly(self, n, x):
        """Recursive formula to compute Chebyshev Polynomials T_n(x)"""
        if n == 0:
            # base case: T_0 = 1
            return torch.ones_like(x)
        elif n == 1:
            # base case: T_1 = x
            return x
        else:
            # recurrence relation for chebyshev polynomials
            # of the first kind  T_{n} = 2*x*T_{n-1} - T_{n-2}
            return torch.mul(
                2 * x, self.chebyshev_poly(n - 1, x)
            ) - self.chebyshev_poly(n - 2, x)

    def forward(self, batch):
        P_m = 0

        if self.include_linear_mixing:
            # linear mixing term
            for i in range(self.n_components):
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
                P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        # predict polynomial coefficients
        concat_embedding = tuple(
            batch[f"embedding_{i}"] for i in range(self.n_components)
        )
        concat_embedding = torch.hstack(concat_embedding)
        coeffients = self.coeffients(concat_embedding)

        # binary excess term
        x_i = batch["composition_0"]
        x_j = batch["composition_1"]
        x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]
        x = torch.abs(1 - 2.0 * x_j)

        for m in range(self.polynomial_order):
            summation = torch.mul(coeffients[:, m], self.chebyshev_poly(m, x))
            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # [batch_size, 1]


class BezierPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,  # = order of Bézier curve (n+1 control points)
        n_components: int = 2,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
        )

        self.parametric_var = torch.tensor(
            [i * (1 / 3) for i in range(self.polynomial_order)]
        ).view(self.polynomial_order, 1)
        embed_dim += 1  # include temperature
        self.mlp_AB = PredictionTaskHead(
            embed_dim=n_components * embed_dim,
        )

        self.mlp_BA = PredictionTaskHead(
            embed_dim=n_components * embed_dim,
        )

        self.berstein_basis = self.compute_basis(self.parametric_var)

    def compute_basis(self, t):
        n = self.polynomial_order
        return torch.hstack([self.bernstein_poly(t, i, n - 1) for i in range(n)])

    def comb(self, n, k):
        n = torch.tensor(n)
        k = torch.tensor(k)
        return torch.exp(
            torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        )

    def bernstein_poly(self, t, i, n):
        """Calculate the Bernstein polynomial of n, i as a part of Bézier."""
        return self.comb(n, i) * (t**i) * ((1 - t) ** (n - i))

    def forward(self, batch):
        P_m = 0

        if self.include_linear_mixing:
            # Linear mixing term
            for i in range(self.n_components):
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
                P_m += torch.mul(batch[f"composition_{i}"].view(-1, 1), P_i)

        # Predict properties fixed parametric points

        embedding_AB = torch.hstack(
            tuple(batch[f"embedding_{i}"] for i in range(self.n_components))
        )
        embedding_BA = torch.hstack(
            tuple(
                batch[f"embedding_{self.n_components - i}"]
                for i in range(1, self.n_components + 1)
            )
        )

        P1 = self.mlp_AB(
            embedding_AB
        )  # Value of property at x = 1/3, Size([batch_size, 1]
        P2 = self.mlp_BA(
            embedding_BA
        )  # Value of property at x = 2/3, Size([batch_size, 1]

        batch_size = embedding_BA.size()[0]
        B = torch.tile(self.berstein_basis, (batch_size, 1, 1))
        P = torch.tile(torch.zeros_like(self.parametric_var), (batch_size, 1, 1))
        P[:, 1, :] = P1
        P[:, 2, :] = P2
        c = torch.linalg.solve(B, P)  # Size([batch_size, polynomial_order, 1])
        c = c.to(embedding_BA.device)  # linalg.solve moves tensor to cpu

        # Binary excess term at input composition
        B_k = self.compute_basis(batch["composition_0"].view(-1, 1)).unsqueeze(
            1
        )  # Shape: (batch_size, 1, n)

        P_m += torch.bmm(B_k, c).squeeze(1)  # Shape: (batch_size, 1, m)

        return P_m  # [batch_size, 1]


class PolynomialHead(Enum):
    """Enumeration of supported polynomial heads."""

    RK = "rk"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    BEZIER = "bezier"

    def get_class(self):
        if self == PolynomialHead.RK:
            return RKPredictionTaskHead
        elif self == PolynomialHead.CHEBYSHEV:
            return ChebyshevPredictionTaskHead
        elif self == PolynomialHead.LEGENDRE:
            return LegendrePredictionTaskHead
        elif self == PolynomialHead.BEZIER:
            return BezierPredictionTaskHead
