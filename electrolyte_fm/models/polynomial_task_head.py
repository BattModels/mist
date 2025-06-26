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
    def forward(self, batch):
        emb_A = batch["embedding_0"]
        emb_B = batch["embedding_1"]
        AB = torch.linalg.cross(emb_A, emb_B)
        BA = torch.linalg.cross(emb_B, emb_A)
        return AB, BA


class DifferenceFusion(nn.Module):
    def forward(self, batch):
        emb_A = batch["embedding_0"]
        emb_B = batch["embedding_1"]
        AB = emb_A - emb_B
        BA = emb_B - emb_A
        return AB, BA


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
        fusion: str = "attention",
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

        self.fusion = CrossAttentionFusion(
            embed_dim - 1, num_heads=num_heads, dropout=dropout, pool="mean"
        )
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
            nn.Linear(embed_dim, 2 * embed_dim),
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
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
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

        batch_size = embedding_BA.size()[0]
        P = torch.tile(torch.zeros_like(self.parametric_var), (batch_size, 1, 1))
        P[:, 1, :] = P1[:, 0].unsqueeze(1)
        P[:, 2, :] = P1[:, -1].unsqueeze(1)
        P[:, 3, :] = P2[:, 0].unsqueeze(1)
        P[:, 4, :] = P2[:, -1].unsqueeze(1)

        B_inv = self.basis_inv.to(P.device)  # (n, n)
        B_inv = B_inv.expand(batch_size, -1, -1)  # batched view
        c = torch.bmm(B_inv, P)  # Shape: (batch_size, n, 1)

        x = batch["composition_0"]
        B_k = self.compute_basis(x.view(-1, 1)).unsqueeze(1)  # (batch_size, 1, n)
        P_m += torch.bmm(B_k, c).squeeze(1)  # (batch_size, 1)

        return P_m


class FusionStrategy(Enum):
    """Enumeration of supported polynomial heads."""

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


class PolynomialHead(Enum):
    """Enumeration of supported polynomial heads."""

    RK = "rk"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    BEZIERFOURTH = "bezier-fourth"

    def get_class(self):
        if self == PolynomialHead.RK:
            return NotImplementedError("TODO: Bring back other polyns")
        elif self == PolynomialHead.CHEBYSHEV:
            raise NotImplementedError("TODO: Bring back other polyns")
        elif self == PolynomialHead.LEGENDRE:
            return NotImplementedError("TODO: Bring back other polyns")
        elif self == PolynomialHead.BEZIERFOURTH:
            return BezierFourthPredictionTaskHead
