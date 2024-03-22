import json
from pathlib import Path
import torch
import pytorch_lightning as pl
from transformers import RobertaConfig, RobertaForMaskedLM
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


class RoBERTa(pl.LightningModule):
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

    def on_train_epoch_start(self) -> None:
        # Update the dataset's internal epoch counter
        self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)
        self.log(
            "train/dataloader_epoch",
            self.trainer.train_dataloader.dataset._epoch,
            rank_zero_only=True,
            sync_dist=True,
        )
        return super().on_train_epoch_start()

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
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
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

    @classmethod
    def load_deepspeed(cls, checkpoint_dir, config_path=None):
        """Restore from a deepspeed checkpoint, mainly used for downstream tasks"""
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config_path = config_path or checkpoint_dir.parent.parent.joinpath(
            "model_hparams.json"
        )
        assert (
            checkpoint_dir.is_dir()
        ), f"Missing deepspeed checkpoint director {checkpoint_dir}"
        assert config_path.is_file(), f"Missing model config file {config_path}"

        # Restore mode from config
        with open(config_path, "r") as fid:
            model_config = json.load(fid)
        model = cls(**model_config)

        # Load model weights from checkpoint
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        model.load_state_dict(state, strict=True, assign=True)
        return model
