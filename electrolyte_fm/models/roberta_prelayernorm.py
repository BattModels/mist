from copy import deepcopy

from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM

from .roberta_base import RoBERTa


class RoBERTaPreLayerNorm(RoBERTa):
    """
    PyTorch Lightning module for RoBERTa model MLM pre-training.
    """

    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        model_kwargs = deepcopy(self.config.to_dict())
        model_kwargs.pop("model_type", None)
        self.config = RobertaPreLayerNormConfig(**model_kwargs)

    def configure_model(self):
        if not hasattr(self, "model"):
            self.model = RobertaPreLayerNormForMaskedLM(config=self.config)
            self._configure_embedding_padding(self.model.roberta_prelayernorm)

    def get_encoder(self):
        if not hasattr(self, "model"):
            self.configure_model()
        return self.model.roberta_prelayernorm
