import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM

from .roberta_base import RoBERTa


class RoBERTaPreLayerNorm(RoBERTa):
    """
    PyTorch Lightning module for RoBERTa model MLM pre-training.
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
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super(LightningModule).__init__()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.vocab_size = vocab_size
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])

        self.config = RobertaPreLayerNormConfig(
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            initializer_range=initializer_range,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )

    def configure_model(self):
        self.model = RobertaPreLayerNormForMaskedLM(config=self.config)
