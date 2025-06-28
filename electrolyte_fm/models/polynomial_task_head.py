from enum import Enum

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.pool = pool.lower()
        if self.pool not in {"mean", "max", "cls"}:
            raise ValueError("pool must be 'mean', 'max', or 'cls'")

    def _pool(self, x, mask=None):
        if self.pool == "cls":  # use first token
            return x[:, 0]  # Shape: (batch_size, d)

        if mask is not None:  # set pads to −inf / 0
            if self.pool == "max":
                x = x.masked_fill(mask.unsqueeze(-1), -float("inf"))
            else:  # mean
                x = x.masked_fill(mask.unsqueeze(-1), 0)

        if self.pool == "max":
            return x.max(dim=1).values  # Shape: (batch_size, d)

        # mean pooling
        if mask is None:
            return x.mean(dim=1)
        lens = (~mask).sum(dim=1, keepdim=True)
        return x.sum(dim=1) / lens.clamp(min=1)

    def forward(self, batch):
        A_tokens, B_tokens = batch["tokens_0"], batch["tokens_1"]
        A_padmask = batch.get("padmask_0")  # may be None
        B_padmask = batch.get("padmask_1")
        AB_ctx, _ = self.cross(
            query=A_tokens, key=B_tokens, value=B_tokens, key_padding_mask=B_padmask
        )

        BA_ctx, _ = self.cross(
            query=B_tokens, key=A_tokens, value=A_tokens, key_padding_mask=A_padmask
        )

        pooled_AB = self._pool(AB_ctx, A_padmask)  # Shape: (batch_size, d)
        pooled_BA = self._pool(BA_ctx, B_padmask)
        return pooled_AB, pooled_BA


class CrossProductFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.down_projection = nn.ModuleList(
            nn.Linear(self.head_dim, 3, bias=False) for _ in range(self.num_heads)
        )
        self.up_projection = nn.ModuleList(
            nn.Linear(3, self.head_dim, bias=False) for _ in range(self.num_heads)
        )

    def _pair_cross(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pieces = []
        for h in range(self.num_heads):
            a3 = self.down_projection[h](a[:, h])  # (B, 3)
            b3 = self.down_projection[h](b[:, h])  # (B, 3)
            c3 = torch.cross(a3, b3, dim=-1)  # (B, 3)
            pieces.append(self.up_projection[h](c3).unsqueeze(1))  # (B,1, D)
        return torch.cat(pieces, dim=1)  # (B, H,  D)

    def forward(self, batch):
        emb_A, emb_B = batch["embedding_0"], batch["embedding_1"]  # (B, D)
        B, D = emb_A.shape
        H, d = self.num_heads, self.head_dim
        emb_A = emb_A.reshape(B, H, d)
        emb_B = emb_B.reshape(B, H, d)
        AB = self._pair_cross(emb_A, emb_B).reshape(B, D)
        BA = -1 * AB
        return AB, BA


class DifferenceFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()

    def forward(self, batch):
        emb_A = batch["embedding_0"]
        emb_B = batch["embedding_1"]
        AB = emb_A - emb_B
        BA = emb_B - emb_A
        return AB, BA


class FusionStrategy(Enum):
    """Enumeration of supported embedding fusion strategies."""

    ATTENTION = "attention"
    CROSSPRODUCT = "cross_product"
    DIFFERENCE = "difference"

    def get_class(self):
        if self == FusionStrategy.ATTENTION:
            return CrossAttentionFusion
        elif self == FusionStrategy.CROSSPRODUCT:
            return CrossProductFusion
        elif self == FusionStrategy.DIFFERENCE:
            return DifferenceFusion


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


class BezierFourthPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        num_heads: int = 1,
        n_components: int = 2,
        include_linear_mixing: bool = False,
        dropout: float = 0.2,
        fusion: str | FusionStrategy = FusionStrategy.ATTENTION,
    ) -> None:
        assert (
            polynomial_order == 4
        ), "BezierCurve only supports 6 control points, i.e 4th order"

        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

        self.fusion_strategy = FusionStrategy(fusion)
        fusion_args = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "dropout": 0.1,
            "pool": "mean",
        }

        self.fusion = self.fusion_strategy.get_class()(**fusion_args)

        n_control_points = polynomial_order + 2
        pi = torch.acos(torch.zeros(1)) * 2
        chebyshev_nodes = torch.tensor(
            [
                torch.cos((2 * k + 1) * pi / (2 * n_control_points))
                for k in range(n_control_points)
            ][::-1]
        ).view(n_control_points, 1)
        self.scale = chebyshev_nodes.max() - chebyshev_nodes.min()
        self.shift = chebyshev_nodes.min()
        self.parametric_var = (chebyshev_nodes - self.shift) / self.scale

        # pre-compute basis & its inverse
        basis = self.compute_basis(self.parametric_var)  # (n, n)
        basis_inv = torch.linalg.inv(basis)  # (n, n)

        # keep on buffer so it moves with .to(device) / .cuda()
        self.register_buffer("basis_inv", basis_inv)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 2 * embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

    def compute_basis(self, t):
        n = self.polynomial_order + 2
        return torch.hstack([self.chebyshev_poly(t, i) for i in range(n)])

    def chebyshev_poly(self, t, i):
        """Chebyshev polynomials of the first kind."""
        return torch.cos(i * torch.arccos(t * self.scale + self.shift))

    def forward(self, batch):
        P_m = 0
        if self.include_linear_mixing:
            for i in range(self.n_components):
                single_emb_with_temp = torch.hstack(
                    (batch["temperature"].view(-1, 1), batch[f"embedding_{i}"])
                ).float()
                P_i = self.single_substance_property(single_emb_with_temp)
                P_m += batch[f"composition_{i}"].view(-1, 1) * P_i

        embedding_AB, embedding_BA = self.fusion(batch)

        # Concantenate temperature to both embeddings
        embedding_AB = torch.hstack(
            (batch["temperature"].view(-1, 1), embedding_AB)
        ).float()
        embedding_BA = torch.hstack(
            (batch["temperature"].view(-1, 1), embedding_BA)
        ).float()

        # Property values at 2 control points closer to x_A = 0
        P1 = self.mlp(embedding_AB)
        # Property values at 2 control points closer to x_B = 0
        P2 = self.mlp(embedding_BA)

        if self.fusion_strategy is FusionStrategy.DIFFERENCE:
            P1 = self.mlp(embedding_AB) - self.mlp(torch.zeros_like(embedding_AB))
            P2 = self.mlp(embedding_BA) - self.mlp(torch.zeros_like(embedding_BA))

        batch_size = embedding_BA.size()[0]
        P = torch.tile(torch.zeros_like(self.parametric_var), (batch_size, 1, 1))
        P[:, 1, :] = P1[:, 0].unsqueeze(1)
        P[:, 2, :] = P1[:, -1].unsqueeze(1)
        P[:, 3, :] = P2[:, 0].unsqueeze(1)
        P[:, 4, :] = P2[:, -1].unsqueeze(1)

        B_inv = self.basis_inv.to(P.device)  # (n, n)
        B_inv = B_inv.expand(batch_size, -1, -1)  # batched view
        c = torch.bmm(B_inv, P).to(embedding_BA.device)  # Shape: (batch_size, n, 1)

        x = batch["composition_0"]
        B_k = self.compute_basis(x.view(-1, 1)).unsqueeze(1)  # (batch_size, 1, n)
        P_m += torch.bmm(B_k, c).squeeze(1)  # (batch_size, 1)

        return P_m


class RKPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        n_components: int = 2,
        include_linear_mixing: bool = False,
        **kwargs,  # allow used kwargs for compatibility between task heads
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

    def forward(self, batch):
        P_m = 0

        for i in range(self.n_components):
            single_emb_with_temp = torch.hstack(
                (batch["temperature"].view(-1, 1), batch[f"embedding_{i}"])
            ).float()
            # non-fusion heads (all except BezierFourthPredictionTaskHead)
            # have temperature concantenated to molecule embedding
            batch[f"embedding_{i}"] = single_emb_with_temp

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
        **kwargs,  # allow used kwargs for compatibility between task heads
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

        for i in range(self.n_components):
            single_emb_with_temp = torch.hstack(
                (batch["temperature"].view(-1, 1), batch[f"embedding_{i}"])
            ).float()
            # non-fusion heads (all except BezierFourthPredictionTaskHead)
            # have temperature concantenated to molecule embedding
            batch[f"embedding_{i}"] = single_emb_with_temp

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
        **kwargs,  # allow used kwargs for compatibility between task heads
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

        for i in range(self.n_components):
            single_emb_with_temp = torch.hstack(
                (batch["temperature"].view(-1, 1), batch[f"embedding_{i}"])
            ).float()
            # non-fusion heads (all except BezierFourthPredictionTaskHead)
            # have temperature concantenated to molecule embedding
            batch[f"embedding_{i}"] = single_emb_with_temp

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


class PolynomialHead(Enum):
    """Enumeration of supported polynomial heads."""

    RK = "rk"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    BEZIERFOURTH = "bezier-fourth"

    def get_class(self):
        if self == PolynomialHead.RK:
            return RKPredictionTaskHead
        elif self == PolynomialHead.CHEBYSHEV:
            return ChebyshevPredictionTaskHead
        elif self == PolynomialHead.LEGENDRE:
            return LegendrePredictionTaskHead
        elif self == PolynomialHead.BEZIERFOURTH:
            return BezierFourthPredictionTaskHead
