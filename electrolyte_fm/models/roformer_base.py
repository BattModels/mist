import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from transformers import RoFormerConfig, RoFormerForMaskedLM

from .roberta_base import RoBERTa


class RoFormer(RoBERTa):
    """
    PyTorch Lightning module for RoFormer model MLM pre-training.
    """

    def __init__(
        self,
        vocab_size: int,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
        hidden_size: int = 768,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.vocab_size = vocab_size
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])

        self.config = RoFormerConfig(
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )

    def configure_model(self):
        self.model = RoFormerForMaskedLM(config=self.config)

    def get_encoder(self):
        if not hasattr(self, "model"):
            self.configure_model()
        return self.model.roformer
