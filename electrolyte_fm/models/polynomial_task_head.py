import math
import torch
from torch import nn


class PolynomialPredictionTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
    ) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.n_components = n_components
        embed_dim += 1  # Temperature appended to embedding

        # predict property for single substance in mixture (P_i)
        self.single_substance_property = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.coeffients = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, polynomial_order),
        )

    def forward(self, batch):
        raise NotImplementedError


class RKPredictionTaskHead(PolynomialPredictionTaskHead):
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

    def forward(self, batch):
        P_m = 0

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
                for k in range(self.polynomial_order):
                    x_ix_j = torch.mul(x_i, x_j)  # [batch_size, 1]
                    RK_summation = torch.mul(
                        torch.mul((-1.0) ** k, RK_coeffients[:, k]),
                        torch.pow((x_i - x_j), k),
                    )
                    P_m += torch.mul(x_ix_j, RK_summation).view(-1, 1)

        return P_m  # [batch_size, 1]


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

        x = 1 - 2 * x_j

        for m in range(self.polynomial_order):
            summation = torch.mul(
                torch.mul((-1.0) ** m, coeffients[:, m]),
                self.legendre_poly(m, x),
            )

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
        x = 1 - 2.0 * x_j

        for m in range(self.polynomial_order):
            summation = torch.mul(
                torch.mul((-1.0) ** m, coeffients[:, m]),
                self.chebyshev_poly(m, x),
            )

            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # [batch_size, 1]
