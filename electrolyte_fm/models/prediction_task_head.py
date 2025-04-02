from typing import Optional
import torch
from torch import nn


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
        num_ref_dist: int = 0,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.n_ref_dist = num_ref_dist
        self.interaction = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_ratio * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        num_pw_dist = self.num_attention_heads + num_ref_dist
        self.distance1 = nn.Sequential(
            nn.Linear(num_pw_dist, num_pw_dist * ff_ratio),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(num_pw_dist * ff_ratio, num_pw_dist),
        )
        self.distance = nn.Sequential(
            nn.Linear(num_pw_dist, num_pw_dist),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(num_pw_dist, 1, bias=False),
        )

    def forward(
        self, hs: torch.Tensor, ref_dist: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, _ = hs.shape
        hs = self.interaction(hs)

        # Multi-head feature distance
        H = self.num_attention_heads
        xb = hs.reshape(B, S, H, -1).transpose(-3, -2)
        d = torch.cdist(xb, xb, compute_mode="use_mm_for_euclid_dist").transpose(-3, -1)

        # Concat Ref Distances
        if self.n_ref_dist > 0:
            assert ref_dist is not None and ref_dist.shape[0:3] == (B, S, S)
            ref_dist = ref_dist if ref_dist.ndim == 4 else ref_dist.unsqueeze(-1)
            d = torch.cat((d, ref_dist.to_dense().to(d)), dim=-1)

        # Pairwise token distances
        d = self.distance1(d) + d
        pw = self.distance(d).squeeze(-1)
        return pw
