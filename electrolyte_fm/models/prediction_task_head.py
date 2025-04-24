import math
import torch
from torch import nn
import torch.nn.functional as F


class PredictionTaskHead(nn.Module):
    def __init__(
        self, embed_dim: int, output_size: int = 1, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.desc_skip_connection = True

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(embed_dim, output_size)

    def forward(self, emb):
        if emb.ndim > 2:
            emb = emb[:, 0, :]
        x_out = self.fc1(emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)
        return z


class TokenTaskHead(nn.Module):
    def __init__(
        self, embed_dim: int, output_size: int = 1, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embed_dim, output_size),
        )

    def forward(self, emb):
        return self.layers(emb)


class TokenPairwiseDistance(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.2,
        num_attention_heads: int = 1,
        activation: str = "relu",
        ff_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.interaction = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_ratio * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.pairwise_distance = nn.Sequential(
            BiPairwiseBlock(embed_dim), nn.Dropout(dropout)
        )
        self.distance1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout), nn.GELU()
        )
        self.distance2 = nn.Sequential(nn.Linear(embed_dim, 1), nn.ReLU())

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.interaction(hs)
        pw_dist = self.pairwise_distance(hs)
        dist = self.distance1(pw_dist) + pw_dist
        return self.distance2(dist).squeeze(-1)


class BiPairwiseBlock(nn.Module):
    def __init__(self, d_model: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.bi_weight = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.lin_weight = nn.Parameter(
            torch.empty((d_model, d_model), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Gradient hook to enforce symmetry
        self.bi_weight.register_hook(lambda grad: 0.5 * (grad + grad.T))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lin_weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.bi_weight, gain=nn.init.calculate_gain("relu"))
        with torch.no_grad():
            self.bi_weight.copy_(0.5 * (self.bi_weight + self.bi_weight.T))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.bias.size(0))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        y_bi = torch.einsum("...ld,df,...rf->...lrf", x, self.bi_weight, x)
        y_bi = 0.5 * (y_bi + y_bi.transpose(-3, -2))  # Enforce symmetry

        x_linear = x.unsqueeze(-2) + x.unsqueeze(-3)
        return y_bi + F.linear(x_linear, self.lin_weight, self.bias)
