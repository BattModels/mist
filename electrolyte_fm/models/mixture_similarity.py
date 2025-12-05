from typing import List, Optional
from itertools import chain

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn

from .mixture_model import MixtureModel


class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate variable-length component embeddings."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Learnable query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size = x.shape[0]

        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, embed_dim)

        # Convert mask for MultiheadAttention (inverted: True=ignore, False=keep)
        if mask is not None:
            key_padding_mask = ~mask  # (B, N)
        else:
            key_padding_mask = None

        # Apply attention: query attends to all components
        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        return pooled.squeeze(1)  # (B, embed_dim)


class MixtureSimilarityModel(MixtureModel):
    """
    PyTorch Lightning module for computing similarity between two mixtures.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        num_heads: int = 8,
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
        # Initialize parent with output_size=1 for similarity prediction
        # Note: output_size and n_components from args are ignored
        super().__init__(
            output_size=1,  # Always 1 for similarity prediction
            encoder_ckpt=encoder_ckpt,
            freeze_encoder=freeze_encoder,
            dropout=dropout,
            vocab_size=vocab_size,
            metrics=metrics,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            transform=transform,
            target_columns=target_columns,
            n_components=0,  # Not used, but required by parent
            temperature=False,  # No temperature dependency
        )

        self.num_heads = num_heads
        embed_dim = self.encoder.config.hidden_size

        # Remove the task_network from parent (not needed for similarity)
        del self.task_network

        # Add attention pooling for each mixture
        self.pool_mix1 = AttentionPooling(embed_dim, num_heads, dropout)
        self.pool_mix2 = AttentionPooling(embed_dim, num_heads, dropout)

    def encode_mixture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a mixture with variable numebrt of components.

        Args:
            input_ids: (B, N, seq_len) - Tokenized SMILES
            attention_mask: (B, N, seq_len) - Attention mask for tokens
            component_mask: (B, N) - Mask for valid components

        Returns:
            Component embeddings: (B, N, embed_dim)
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

        mix1_embedding = self.pool_mix1(mix1_components, batch["component_mask_mix1"])
        mix2_embedding = self.pool_mix2(mix2_components, batch["component_mask_mix2"])

        similarity = F.cosine_similarity(
            mix1_embedding, mix2_embedding, dim=-1, eps=1e-8
        )
        similarity = similarity.unsqueeze(-1)  # (B, 1)

        if transform:
            similarity = self.transform.forward(similarity)

        mixture_embeddings = {
            "mix1_embedding": mix1_embedding,
            "mix2_embedding": mix2_embedding,
        }

        return similarity, mixture_embeddings

    def configure_optimizers(self):
        if self.freeze_encoder:
            # Only train pooling layers
            learnable_params = chain(
                self.pool_mix1.parameters(),
                self.pool_mix2.parameters(),
            )
        else:
            # Train encoder + pooling layers
            learnable_params = chain(
                self.encoder.parameters(),
                self.pool_mix1.parameters(),
                self.pool_mix2.parameters(),
            )

        optimizer = self.optimizer(learnable_params)

        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
