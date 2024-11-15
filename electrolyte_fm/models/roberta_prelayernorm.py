import torch
from copy import deepcopy
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM

from .roberta_base import RoBERTa


class RoBERTaPreLayerNorm(RoBERTa):
    """
    PyTorch Lightning module for RoBERTa model MLM pre-training.
    """

    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        model_kwargs = deepcopy(kwargs)
        model_kwargs.pop("lr_schedule", None)
        model_kwargs.pop("optimizer", None)
        self.config = RobertaPreLayerNormConfig(vocab_size=vocab_size, **model_kwargs)

    def configure_model(self):
        self.model = RobertaPreLayerNormForMaskedLM(config=self.config)

    def get_encoder(self):
        if not hasattr(self, "model"):
            self.configure_model()
        return self.model.roberta_prelayernorm
