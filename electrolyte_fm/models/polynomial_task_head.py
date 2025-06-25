from enum import Enum

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """
    Permutation-invariant fusion of two sequences using bidirectional
    cross-attention followed by pooling and symmetric averaging.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.cross_ab = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_ba = nn.MultiheadAttention(
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

    def forward(
        self,
        A_tokens: torch.Tensor,  # Shape: (batch_size, seq_len_A, d)
        B_tokens: torch.Tensor,  # Shape: (batch_size, seq_len_B, d)
        A_padmask: torch.Tensor = None,  # Shape: (batch_size, seq_len_A)
        B_padmask: torch.Tensor = None,
    ):
        AB_ctx, _ = self.cross_ab(
            query=A_tokens, key=B_tokens, value=B_tokens, key_padding_mask=B_padmask
        )

        BA_ctx, _ = self.cross_ba(
            query=B_tokens, key=A_tokens, value=A_tokens, key_padding_mask=A_padmask
        )

        pooled_A = self._pool(AB_ctx, A_padmask)  # Shape: (batch_size, d)
        pooled_B = self._pool(BA_ctx, B_padmask)
        pair_embed = 0.5 * (pooled_A + pooled_B)  # Shape: (batch_size, d)
        return pair_embed


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
        x_ix_j = torch.mul(x_i, x_j)  # Shape: (batch_size, 1)

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
        x_ix_j = torch.mul(x_i, x_j)  # Shape: (batch_size, 1)
        x = torch.abs(1 - 2.0 * x_j)

        for m in range(self.polynomial_order):
            summation = torch.mul(coeffients[:, m], self.chebyshev_poly(m, x))
            P_m += torch.mul(x_ix_j, summation).view(-1, 1)

        return P_m  # Shape: (batch_size, 1)


class BezierPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,  # = order of Bézier curve (n+1 control points)
        n_components: int = 2,
        include_linear_mixing: bool = False,
        dropout: float = 0.2,
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
            nn.Linear(self.n_components * embed_dim, 2 * embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
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

        return P_m  # Shape: (batch_size, 1)


class BezierFourthPredictionTaskHead(PolynomialPredictionTaskHead):
    def __init__(
        self,
        embed_dim: int,
        polynomial_order: int = 4,
        num_heads: int = 1,
        n_components: int = 2,
        include_linear_mixing: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            polynomial_order=polynomial_order,
            n_components=n_components,
            include_linear_mixing=include_linear_mixing,
        )

        self.fusion = CrossAttentionFusion(
            embed_dim, num_heads=num_heads, dropout=dropout, pool="mean"
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

        # --- pre-compute basis & its inverse ------------------------
        basis = self.compute_basis(self.parametric_var)  # (n, n)
        basis_inv = torch.linalg.inv(basis)  # (n, n)

        # keep on buffer so it moves with .to(device) / .cuda()
        self.register_buffer("basis_inv", basis_inv)  # ❶

        # ------------- MLP definition (unchanged except T concat note)
        embed_dim += 1  # +1 for temperature scalar
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, polynomial_order - 2),
        )

    def compute_basis(self, t):
        n = self.polynomial_order
        return torch.hstack([self.chebyshev_poly(t, i) for i in range(n)])

    def chebyshev_poly(self, t, i):
        # Chebyshev polynomials of the first kind
        return torch.cos(i * torch.arccos(t * self.scale + self.shift))

    def forward(self, batch):
        P_m = 0
        # (optional) linear mixing ...................................
        if self.include_linear_mixing:
            for i in range(self.n_components):
                P_i = self.single_substance_property(batch[f"embedding_{i}"])
                P_m += batch[f"composition_{i}"].view(-1, 1) * P_i

        A_tok, B_tok = batch["tokens_0"], batch["tokens_1"]
        A_mask = batch.get("padmask_0")  # may be None
        B_mask = batch.get("padmask_1")

        pair_emb = self.fusion(A_tok, B_tok, A_mask, B_mask)  # (B,d)
        pair_emb = torch.hstack((batch["temperature"].view(-1, 1), pair_emb)).float()
        controls_pts = self.mlp(pair_emb)

        batch_size = pair_emb.size(0)
        n = self.polynomial_order

        # Build P  (shape: batch × n × 1)
        P = torch.zeros(batch_size, n, 1, device=pair_emb.device)
        P[:, 1, 0] = controls_pts[:, 0]  # C₁
        P[:, 2, 0] = controls_pts[:, 1]  # C₂
        P[:, 3, 0] = controls_pts[:, 2]  # C₃
        P[:, 4, 0] = controls_pts[:, 3]  # C₄   (if n == 5)

        B_inv = self.basis_inv.to(P.device)  # (n, n)
        B_inv = B_inv.expand(batch_size, -1, -1)  # batched view
        c = torch.bmm(B_inv, P)  # (batch, n, 1)

        x = batch["composition_0"]
        B_k = self.compute_basis(x.view(-1, 1)).unsqueeze(1)  # (batch, 1, n)
        P_m += torch.bmm(B_k, c).squeeze(1)  # (batch, 1)

        return P_m


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
