from typing import Optional

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from transformers import PreTrainedTokenizerBase


class LLMPredictions(Callback):
    def __init__(self, every_n_steps=10):
        self.every_n_steps = every_n_steps
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "L.LightningModule"):
        dm = trainer.datamodule
        assert hasattr(dm, "tokenizer")
        assert isinstance(dm.tokenizer, PreTrainedTokenizerBase)
        self.tokenizer = dm.tokenizer

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_steps != 0:
            return
        token_ids = outputs["logits"].argmax(dim=-1)
        mask = batch["labels"] != -100
        for bdx in range(token_ids.shape[0]):
            source = self.tokenizer.decode(
                batch["labels"][bdx, mask[bdx, :]], skip_special_tokens=False
            )
            text = self.tokenizer.decode(
                token_ids[bdx, mask[bdx, :].flatten()], skip_special_tokens=False
            )
            print(f"{source} => {text}")
