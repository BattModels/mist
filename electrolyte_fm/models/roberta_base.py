import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.loggers import WandbLogger
from transformers import RobertaConfig, RobertaForMaskedLM

from .model_utils import DeepSpeedMixin, LoggingMixin

from mup import make_base_shapes, set_base_shapes, MuAdamW


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
        base_intermediate_size: int = 256,
        base_num_attention_heads: int = 4,
        base_num_hidden_layers: int = 3,
        base_hidden_size: int = 256,
        delta_intermediate_size: int = 300,
        delta_num_attention_heads: int = 5,
        delta_num_hidden_layers: int = 2,
        delta_hidden_size: int = 200,
        optimizer: OptimizerCallable = MuAdamW,
        base_shape_save_path:str = 'base_shape/bert256.bsh',
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.vocab_size = vocab_size
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])

        # Define base and delta model
        self.base_config = RobertaConfig(
            vocab_size=vocab_size,
            intermediate_size=base_intermediate_size,
            hidden_size=base_hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=base_num_attention_heads,
            num_hidden_layers=base_num_hidden_layers,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )
        self.base_model = RobertaForMaskedLM(config=self.base_config)

        self.delta_config = RobertaConfig(
            vocab_size=vocab_size,
            intermediate_size=delta_intermediate_size,
            hidden_size=delta_hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=delta_num_attention_heads,
            num_hidden_layers=delta_num_hidden_layers,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )
        self.delta_model = RobertaForMaskedLM(config=self.delta_config)

        # define a base shape object based on comparing delta_model against base_model
        base_shapes = make_base_shapes(self.base_model, self.delta_model, savefile=base_shape_save_path)

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

        # set base shapes
        set_base_shapes(self.model, base_shapes)

        # re-initialize
        self.model.apply(self.model._init_weights)

    def get_encoder(self):
        return self.model.roberta

    def forward(self, batch, **kwargs):  # type: ignore[override]
        out = self.model(
            batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )
        return out

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

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
        optimizer = self.optimizer(self.parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
