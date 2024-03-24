import json
from pathlib import Path
import torch
import pytorch_lightning as pl
from electrolyte_fm.models import RoBERTa, ClassificationHead
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


class RoBERTaClassification(pl.LightningModule):
    """
    PyTorch Lightning module for RoBERTa model classification finetuning.
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        learning_rate: float = 1.6e-4,
        num_classes: int = 1, 
        dropout: float = 0.2

    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.save_hyperparameters()
        self.pretrained_model = RoBERTa.load_deepspeed(checkpoint_dir= pretrained_checkpoint)
        # Expose encoder
        self.encoder = self.pretrained_model.model.roberta
        self.task_network = ClassificationHead(embed_dim=self.encoder.config.hidden_size, 
                                               num_classes=num_classes,
                                               dropout=self.dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch, **kwargs):  # type: ignore[override]
        embedding = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )
        out = self.task_network(embedding.last_hidden_state)
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
        loss = self.loss(outputs, targets)
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
        loss = self.loss(outputs, targets)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        # loss = self.loss(outputs, targets)
        # self.log(
        #     "test/loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        return outputs

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
        print("checkpoint_dir.parent.parent")
        print(checkpoint_dir.parent.parent)
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
