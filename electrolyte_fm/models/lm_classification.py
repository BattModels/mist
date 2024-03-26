import json
from pathlib import Path
import torch
import pytorch_lightning as pl
from electrolyte_fm.models import (
    ClassificationHead, DeepSpeedMixin, LoggingMixin
)
class LMClassification(LoggingMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on classification 
    tasks.
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        encoder_class: str = "roberta",
        freeze_encoder: bool = True,
        learning_rate: float = 1.6e-4,
        num_classes: int = 2, 
        dropout: float = 0.2

    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoder_class = encoder_class
        self.save_hyperparameters()
        self.pretrained_model = DeepSpeedMixin.load_deepspeed(
            encoder_class = self.encoder_class,
            checkpoint_dir= pretrained_checkpoint
            )
        # Expose encoder
        self.encoder = self.pretrained_model.model.roberta
        self.task_network = ClassificationHead(embed_dim=self.encoder.config.hidden_size, 
                                               num_classes=num_classes,
                                               dropout=self.dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.freeze_encoder = freeze_encoder

    def forward(self, batch, **kwargs):  # type: ignore[override]
        embedding = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
            return_dict=False
        )
        
        sequence_output = embedding[0]
        out = self.task_network(sequence_output)
        return out

    def configure_optimizers(self):
        learnable_params = [p for _, p in self.task_network.named_parameters()]
        if not self.freeze_encoder:
            learnable_params.extend([p for _, p in self.encoder.named_parameters()])
        optimizer = torch.optim.AdamW(
            learnable_params,
            lr=self.learning_rate,
        )
        return optimizer
