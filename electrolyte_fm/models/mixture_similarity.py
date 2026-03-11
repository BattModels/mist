import math
from typing import List, Optional
from itertools import chain

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn

from .mixture_model import MixtureModel
from .normalize import AbstractNormalizer


class R2Loss(nn.Module):
    """R2-based loss function: loss = 1 - R²"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.flatten()
        targets = targets.flatten()

        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return 1 - r2


class EmbeddingDistance(nn.Module):
    """
    Evaluate distance between embeddings and map to a [0, 1] range:
    where 0 is total overlap, 1 is the furthest away.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance between embeddings"""
        raise NotImplementedError

    @classmethod
    def get(cls, distance: str, **kwargs) -> "EmbeddingDistance":
        if distance in ["cosine", CosineDistance.__name__]:
            return CosineDistance(**kwargs)
        elif distance in ["scaled_cosine", ScaledCosineDistance.__name__]:
            return ScaledCosineDistance(**kwargs)
        elif distance in ["euclidean", EuclideanDistance.__name__]:
            return EuclideanDistance(**kwargs)
        elif distance in ["manhattan", ManhattanDistance.__name__]:
            return ManhattanDistance(**kwargs)


class EuclideanDistance(EmbeddingDistance):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1n = F.normalize(x1, p=2, dim=1, eps=self.eps)
        x2n = F.normalize(x2, p=2, dim=1, eps=self.eps)

        d = torch.linalg.norm(x1n - x2n, dim=1)  # (B,), in [0, 2]
        y = (d * 0.5).unsqueeze(-1)  # (B, 1), in [0, 1]
        return y


class ManhattanDistance(EmbeddingDistance):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1n = F.normalize(x1, p=2, dim=1, eps=self.eps)
        x2n = F.normalize(x2, p=2, dim=1, eps=self.eps)

        d1 = (x1n - x2n).abs().sum(dim=1)  # (B,), in [0, 2*sqrt(D)]
        D = x1n.size(1)
        y = (d1 / (2.0 * math.sqrt(D))).unsqueeze(-1)  # (B, 1), in [0, 1]
        return y


class CosineDistance(EmbeddingDistance):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=1, eps=self.eps)  # (B,)
        return ((1.0 - cos) * 0.5).unsqueeze(-1)  # (B, 1)


class ScaledCosineDistance(EmbeddingDistance):
    def __init__(self, out_dim: int = 1):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.scaler = nn.Linear(1, 1, bias=True)

        with torch.no_grad():
            self.scaler.weight.fill_(6.0)
            self.scaler.bias.fill_(-6.0)

    def forward(self, x1, x2):
        cos = self.cosine(x1, x2).unsqueeze(-1)  # (B, 1),
        d = 1 - cos  # (B, 1)
        return torch.sigmoid(self.scaler(d))  # (B, 1)


class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate variable-length component embeddings."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size = x.shape[0]

        query = self.query.expand(batch_size, -1, -1)  # (B, 1, embed_dim)

        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=mask,
        )

        return pooled.squeeze(1)  # (B, embed_dim)


class WeightedSumPooling(nn.Module):
    """Weighted sum pooling where weights are predicted from component embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        # Network to predict weight from embedding
        self.weight_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        logits = self.weight_predictor(x).squeeze(-1)  # (B, N)

        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        weights = torch.softmax(logits, dim=-1)  # (B, N)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        return pooled


class MixtureSimilarityModel(MixtureModel):
    """
    PyTorch Lightning module for computing similarity/ dissimilarity between two mixtures.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        distance_metric: str = "scaled_cosine",
        freeze_encoder: bool = False,
        dropout: float = 0.2,
        vocab_size: Optional[int] = None,
        metrics: List[str] = ["mae", "rmse", "r2"],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str | list[str]] = None,
        target_columns: Optional[List[str]] = None,
        output_size: int = 1,  # Ignored, always 1 for similarity
        n_components: int = 0,  # Ignored, for config compatibility
        **kwargs,
    ) -> None:
        super().__init__(
            output_size=1,
            encoder_ckpt=encoder_ckpt,
            freeze_encoder=freeze_encoder,
            dropout=dropout,
            vocab_size=vocab_size,
            metrics=metrics,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            transform=transform,
            target_columns=target_columns,
            n_components=0,
            temperature=False,
        )

        # Override loss function to use R2 loss
        self.lossfn = R2Loss()

        embed_dim = self.encoder.config.hidden_size
        self.distance_metric = distance_metric

        del self.task_network

        self.pool_mix = WeightedSumPooling(embed_dim, hidden_dim=128, dropout=dropout)
        self.similarity_regressor = EmbeddingDistance.get(self.distance_metric)

    def _scaled_pred_loss(self, batch):
        """Compute R2 loss for mixture similarity."""
        preds, mix_embedding = self.forward(batch, transform=False)
        target = batch["target"]
        target = self.transform.inverse(target)

        mask = batch["target_mask"]
        valid_preds = preds[mask]
        valid_targets = target[mask]

        if valid_preds.numel() > 0:
            loss = self.lossfn(valid_preds, valid_targets)
        else:
            loss = torch.tensor(0.0, device=preds.device)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        preds = self.transform.forward(preds)
        return preds, loss

    def encode_mixture(self, input_ids, attention_mask, component_mask):
        batch_size, max_components, seq_len = input_ids.shape
        input_ids_flat = input_ids.reshape(-1, seq_len)
        attention_mask_flat = attention_mask.reshape(-1, seq_len)

        out = self.encoder(
            input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True,
            output_hidden_states=False,
        )
        last_hidden = out.last_hidden_state  # (B*N, L, d)

        comp_emb = last_hidden[:, 0, :]
        comp_emb = comp_emb.view(batch_size, max_components, -1)

        # If component_mask is False for present, True for missing, zero missing embeddings
        if component_mask is not None:
            comp_emb = comp_emb * (~component_mask).unsqueeze(-1).type_as(comp_emb)

        return comp_emb

    def forward(self, batch, transform=True, **kwargs):  # type: ignore[override]
        mix1_components = self.encode_mixture(
            batch["input_ids_mix1"],
            batch["attention_mask_mix1"],
            batch["component_mask_mix1"],
        )

        mix2_components = self.encode_mixture(
            batch["input_ids_mix2"],
            batch["attention_mask_mix2"],
            batch["component_mask_mix2"],
        )

        mix1_embedding = self.pool_mix(mix1_components, batch["component_mask_mix1"])
        mix2_embedding = self.pool_mix(mix2_components, batch["component_mask_mix2"])

        output = self.similarity_regressor(mix1_embedding, mix2_embedding)  # (B, 1)

        if transform:
            output = self.transform.forward(output)

        mixture_embeddings = {
            "mix1_embedding": mix1_embedding,
            "mix2_embedding": mix2_embedding,
        }
        return output, mixture_embeddings

    def configure_optimizers(self):
        if self.freeze_encoder:
            param_groups = [
                self.pool_mix.parameters(),
            ]
        else:
            param_groups = [
                self.encoder.parameters(),
                self.pool_mix.parameters(),
            ]

        param_groups.append(self.similarity_regressor.parameters())
        learnable_params = chain(*param_groups)
        optimizer = self.optimizer(learnable_params)

        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
