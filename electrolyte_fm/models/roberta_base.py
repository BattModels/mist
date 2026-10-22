import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from transformers import RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizerBase

from electrolyte_fm.utils.tokenizer import load_tokenizer

from ..utils.metrics import TokenCounter
from .model_utils import CanSkip, DeepSpeedMixin, LoggingMixin, create_position_ids


class RoBERTa(LightningModule, DeepSpeedMixin, LoggingMixin, CanSkip):
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
        layer_norm_eps: float = 1e-12,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        enable_token_counter: bool = True,
        tokenizer: str | None = None,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.vocab_size = vocab_size
        self.token_counter = None
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        if enable_token_counter:
            self.token_counter = TokenCounter()
        self.config = RobertaConfig(
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

        if tokenizer:
            self.configure_tokenizer(tokenizer)

    def configure_model(self):
        if not hasattr(self, "model"):
            self.model = RobertaForMaskedLM(config=self.config)
            self._configure_embedding_padding(self.model.roberta)

    def configure_tokenizer(self, tokenizer: str | PreTrainedTokenizerBase):
        tokenizer = (
            load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.config.position_padding_idx = getattr(
            self.config, "position_padding_idx", self.config.pad_token_id
        )
        if hasattr(self, "model") and self.config.vocab_size != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))

        # set the config to match the tokenizer
        self.config.vocab_size = len(tokenizer)
        for token in ["pad_token_id", "bos_token_id", "eos_token_id"]:
            token_id = getattr(tokenizer, token, None)
            if token_id is not None:
                setattr(self.config, token, token_id)
        if hasattr(self, "model"):
            self._configure_embedding_padding(self.model.roberta)

    def _configure_embedding_padding(self, encoder):
        position_padding_idx = getattr(
            self.config, "position_padding_idx", self.config.pad_token_id
        )
        encoder.embeddings.padding_idx = position_padding_idx
        encoder.embeddings.position_embeddings.padding_idx = position_padding_idx
        encoder.embeddings.word_embeddings.padding_idx = self.config.pad_token_id

    def get_encoder(self):
        if not hasattr(self, "model"):
            self.configure_model()
        return self.model.roberta

    def forward(self, batch, **kwargs):  # type: ignore[override]
        if "position_ids" not in kwargs and hasattr(
            self.config, "position_padding_idx"
        ):
            kwargs["position_ids"] = create_position_ids(
                batch["input_ids"],
                self.config.pad_token_id,
                self.config.position_padding_idx,
            )
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
        if self.should_skip():
            loss = 0 * loss

        if self.token_counter:
            self.token_counter.update(batch["attention_mask"], batch["labels"])
            self.log_dict(
                self.token_counter.compute(),
                on_step=True,
                on_epoch=True,
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
