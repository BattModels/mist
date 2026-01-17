from typing import List, Optional
from itertools import chain

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn

from .mixture_model import MixtureModel
from .normalize import AbstractNormalizer


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
        # Convert mask for MultiheadAttention (inverted: True=ignore, False=keep)
        if mask is not None:
            key_padding_mask = ~mask  # (B, N)
        else:
            key_padding_mask = None

        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        return pooled.squeeze(1)  # (B, embed_dim)


class ScaledCosineRegressor(nn.Module):
    """
    Use scaled cosine similarity as similarity regressor.
    """

    def __init__(self, out_dim: int = 1):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.scaler = nn.Linear(1, 1, bias=True)

        with torch.no_grad():
            # Example init: roughly map d ∈ [0, 2] to pre-sigmoid [-2, 2]
            self.scaler.weight.fill_(2.0)
            self.scaler.bias.fill_(-2.0)

    def forward(self, x1, x2):
        cos = self.cosine(x1, x2).unsqueeze(-1)  # (B, 1), ∈ [-1, 1]
        d = 1 - cos  # (B, 1), ∈ [0, 2]
        # Optional: enforce monotonicity
        with torch.no_grad():
            self.scaler.weight.clamp_(min=0)
        return torch.sigmoid(self.scaler(d))  # (B, 1), ∈ (0, 1)


class MixtureSimilarityModel(MixtureModel):
    """
    PyTorch Lightning module for computing similarity/ dissimilarity between two mixtures.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        dropout: float = 0.2,
        num_heads: int = 8,
        vocab_size: Optional[int] = None,
        metrics: List[str] = ["mae", "rmse", "r2"],
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        transform: Optional[str | list[str]] = None,
        target_columns: Optional[List[str]] = None,
        output_size: int = 1,  # Ignored, always 1 for similarity
        n_components: int = 0,  # Ignored, for config compatibility
        descriptor_dim: Optional[
            int
        ] = None,  # If provided, fuse descriptors with embeddings
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

        self.num_heads = num_heads
        self.descriptor_dim = descriptor_dim

        embed_dim = self.encoder.config.hidden_size

        del self.task_network

        self.pool_mix = AttentionPooling(embed_dim, num_heads, dropout)

        # Descriptor processing: project to embedding dimension
        if self.descriptor_dim is not None:
            self.desc_transform = AbstractNormalizer.get(
                "standardize", descriptor_dim
            ).eval()
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, int(descriptor_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(descriptor_dim // 2), embed_dim),
            )

        self.similarity_regressor = ScaledCosineRegressor(out_dim=1)

    def encode_mixture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a mixture with variable numebrt of components.
        """
        batch_size, max_components, seq_len = input_ids.shape

        input_ids_flat = input_ids.view(-1, seq_len)  # (B*N, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)  # (B*N, seq_len)

        encoder_output = self.encoder(
            input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True,
            output_hidden_states=True,
        )

        last_hidden = encoder_output.last_hidden_state  # (B*N, seq_len, embed_dim)

        attention_mask_expanded = attention_mask_flat.unsqueeze(-1)  # (B*N, seq_len, 1)
        sum_embeddings = (last_hidden * attention_mask_expanded).sum(
            dim=1
        )  # (B*N, embed_dim)
        sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B*N, 1)
        component_embeddings = sum_embeddings / sum_mask  # (B*N, embed_dim)

        embed_dim = component_embeddings.shape[-1]
        component_embeddings = component_embeddings.view(
            batch_size, max_components, embed_dim
        )

        return component_embeddings

    def on_fit_start(self):
        """Compute normalization statistics for targets and descriptors."""
        super().on_fit_start()

        if self.descriptor_dim is not None:
            desc_state = None
            if self.global_rank == 0:
                desc_ds = self.trainer.datamodule.descriptor_dataset
                if desc_ds is not None:
                    desc_state = self.desc_transform.fit(desc_ds, name="descriptors")

            desc_state = self.trainer.strategy.broadcast(desc_state)
            self.desc_transform.load_state_dict(desc_state)

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

        # Fuse with descriptors if available (concatenation fusion)
        if self.descriptor_dim is not None and "descriptors_mix1" in batch:
            desc1_norm = self.desc_transform.inverse(batch["descriptors_mix1"])
            desc2_norm = self.desc_transform.inverse(batch["descriptors_mix2"])

            desc1_embedding = self.descriptor_mlp(desc1_norm)  # (B, embed_dim)
            desc2_embedding = self.descriptor_mlp(desc2_norm)  # (B, embed_dim)

            mix1_embedding = torch.cat(
                [mix1_embedding, desc1_embedding], dim=-1
            )  # (B, embed_dim * 2)
            mix2_embedding = torch.cat(
                [mix2_embedding, desc2_embedding], dim=-1
            )  # (B, embed_dim * 2)

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

        if self.descriptor_dim is not None:
            param_groups.append(self.descriptor_mlp.parameters())

        learnable_params = chain(*param_groups)
        optimizer = self.optimizer(learnable_params)

        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
