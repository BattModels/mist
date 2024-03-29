import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from .model_utils import DeepSpeedMixin, LoggingMixin
class RoBERTa(DeepSpeedMixin, LoggingMixin):
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
        learning_rate: float = 1.6e-4,
    ) -> None:
        
        super().__init__()
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.save_hyperparameters()

        self.config = RobertaConfig(
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )
        self.model = RobertaForMaskedLM(config=self.config)

    def forward(self, batch, **kwargs):  # type: ignore[override]
        out = self.model(
            batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )
        return out
    
    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val/loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for _, p in self.model.named_parameters()],
            lr=self.learning_rate,
        )
        return optimizer