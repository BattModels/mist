from enum import Enum

import torch
from torch import nn


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

        self.single_substance_property = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self.coeffients = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
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
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
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
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
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
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

        pi = torch.acos(torch.zeros(1)) * 2
        chebyshev_nodes = torch.tensor(
            [
                torch.cos((2 * k + 1) * pi / (2 * polynomial_order))
                for k in range(polynomial_order)
            ]
        ).view(polynomial_order, 1)
        self.scale = chebyshev_nodes.max() - chebyshev_nodes.min()
        self.shift = 0 - chebyshev_nodes.min()
        self.parametric_var = (chebyshev_nodes + self.shift) / self.scale

        embed_dim += 1  # include temperature

        self.mlp = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self.chebyshev_basis = self.compute_basis(self.parametric_var)

    def compute_basis(self, t):
        n = self.polynomial_order
        return torch.hstack([self.chebyshev_poly(t, i) for i in range(n)])

    def chebyshev_poly(self, t, i):
        # Chebyshev polynomials of the first kind
        return torch.cos(i * torch.arccos(t * self.scale - self.shift))

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

        P1 = self.mlp(embedding_AB)

        P2 = self.mlp(embedding_BA)

        batch_size = embedding_BA.size()[0]
        B = torch.tile(self.chebyshev_basis, (batch_size, 1, 1))
        P = torch.tile(torch.zeros_like(self.parametric_var), (batch_size, 1, 1))
        P[:, 1, :] = P1
        P[:, 2, :] = P2
        c = torch.linalg.solve(B, P)  # Size([batch_size, polynomial_order, 1])
        c = c.to(embedding_BA.device)  # linalg.solve moves tensor to cpu

        # Binary excess term at input composition
        x = batch["composition_0"]
        B_k = self.compute_basis(x.view(-1, 1)).unsqueeze(
            1
        )  # Shape: (batch_size, 1, n)

        P_m += torch.bmm(B_k, c).squeeze(1)  # Shape: (batch_size, 1, m)

        return P_m  # [batch_size, 1]


class BezierFourthPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,  # = order of Bézier curve (n+1 control points)
        n_components: int = 2,
        include_linear_mixing: bool = False,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

        pi = torch.acos(torch.zeros(1)) * 2
        chebyshev_nodes = torch.tensor(
            [
                torch.cos((2 * k + 1) * pi / (2 * polynomial_order))
                for k in range(polynomial_order)
            ][::-1]
        ).view(polynomial_order, 1)
        self.scale = chebyshev_nodes.max() - chebyshev_nodes.min()
        self.shift = chebyshev_nodes.min()
        self.parametric_var = (chebyshev_nodes - self.shift) / self.scale
        embed_dim += 1  # include temperature

        self.mlp = nn.Sequential(
            nn.Linear(self.n_components * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        self.chebyshev_basis = self.compute_basis(self.parametric_var)

    def compute_basis(self, t):
        n = self.polynomial_order
        return torch.hstack([self.chebyshev_poly(t, i) for i in range(n)])

    def chebyshev_poly(self, t, i):
        # Chebyshev polynomials of the first kind
        return torch.cos(i * torch.arccos(t * self.scale + self.shift))

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

        P1 = self.mlp(embedding_AB)

        P2 = self.mlp(embedding_BA)

        batch_size = embedding_BA.size()[0]
        B = torch.tile(self.chebyshev_basis, (batch_size, 1, 1))
        P = torch.tile(torch.zeros_like(self.parametric_var), (batch_size, 1, 1))
        P[:, 1, :] = P1[:, 0].unsqueeze(1)
        P[:, 2, :] = P1[:, -1].unsqueeze(1)
        P[:, 3, :] = P2[:, 0].unsqueeze(1)
        P[:, 4, :] = P2[:, -1].unsqueeze(1)
        c = torch.linalg.solve(B, P)  # Size([batch_size, polynomial_order, 1])
        is_close = torch.allclose(
            B @ c,
            P,
            atol=1e-05,
        )
        assert is_close
        c = c.to(embedding_BA.device)  # linalg.solve moves tensor to cpu

        # Binary excess term at input composition
        x = batch["composition_0"]
        B_k = self.compute_basis(x.view(-1, 1)).unsqueeze(
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
    BEZIERFOURTH = "bezier-fourth"

    def get_class(self):
        if self == PolynomialHead.RK:
            return RKPredictionTaskHead
        elif self == PolynomialHead.CHEBYSHEV:
            return ChebyshevPredictionTaskHead
        elif self == PolynomialHead.LEGENDRE:
            return LegendrePredictionTaskHead
        elif self == PolynomialHead.BEZIER:
            return BezierPredictionTaskHead
        elif self == PolynomialHead.BEZIERFOURTH:
            return BezierFourthPredictionTaskHead
