import torch
from torch import nn


def pairwise_fusion(name: str, *args, **kwargs):
    if name == "square-difference":
        return PairwiseInteraction(*args, **kwargs)
    elif name == "gaussian":
        return GaussianFusion(*args, **kwargs)
    elif name == "difference":
        return EquivariantInteraction(*args, **kwargs)
    elif name == "softmax":
        return SoftmaxFusion(*args, **kwargs)
    elif name == "concat":
        return ConcatFusion(*args, **kwargs)
    else:
        raise ValueError(f"Unknown fusion: {name}")


class PairwiseInteraction(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_targets: int = 1,
        dropout: float = 0.1,
        n_env: int = 0,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_targets = n_targets
        self.n_env = n_env
        self.mlp_emb = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_in + n_env, n_in),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(n_in, n_out * n_targets),
        )
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Reset weights of last
        last = self.mlp[-1]
        nn.init.xavier_normal_(last.weight, gain=nn.init.calculate_gain("linear"))

        self.mlp.apply(init_weights)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, e: torch.Tensor | None = None
    ) -> torch.Tensor:
        a = self.mlp_emb(a)
        b = self.mlp_emb(b)
        d = self.distance(a, b)
        if self.n_env > 0:
            d = torch.cat([d, e], dim=-1)
        y = self.mlp(d)
        y = y.reshape(*y.shape[:-1], self.n_targets, self.n_out)
        return y + y.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return (a - b).pow(2)


class GaussianFusion(PairwiseInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = -1 * (a - b).pow(2)
        return dist.exp()


class EquivariantInteraction(PairwiseInteraction):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, e: torch.Tensor | None = None
    ) -> torch.Tensor:
        a = self.mlp_emb(a)
        b = self.mlp_emb(b)
        emb_a, emb_b = self.distance(a, b)
        if self.n_env > 0:
            emb_a = torch.cat([emb_a, e], dim=-1)
            emb_b = torch.cat([emb_b, e], dim=-1)
        y_a = self.mlp(emb_a).reshape(*emb_a.shape[:-1], self.n_targets, self.n_out)
        y_b = self.mlp(emb_b).reshape(*emb_b.shape[:-1], self.n_targets, self.n_out)
        return y_a + y_b.flip(-1)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return a - b, b - a


class SoftmaxFusion(EquivariantInteraction):
    def distance(self, a: torch.Tensor, b: torch.Tensor):
        y = torch.stack([a, b]).log_softmax(dim=0)
        return y[0], y[1]


class ConcatFusion(EquivariantInteraction):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        super().__init__(n_in, n_out, **kwargs)
        self.mlp[0] = nn.Linear(2 * self.n_in + self.n_env, self.n_in)
        self.reset_parameters()

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        return torch.cat([a, b], dim=-1), torch.cat([b, a], dim=-1)
