from itertools import chain
from pathlib import Path
from typing import List, Optional

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger

from ..utils.metrics import get_metrics
from ..utils.tokenizer import load_tokenizer
from .model_utils import DeepSpeedMixin, LoggingMixin
from .physics_task_heads import VFTDecayTaskHead


class IonicConductivityModel(LightningModule, DeepSpeedMixin, LoggingMixin):
    """
    PyTorch Lightning module for mixture ionic conductivity  prediction.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        freeze_encoder: bool = False,
        tokenizer: Optional[str] = None,
        vocab_size: Optional[int] = None,
        dropout: float = 0.1,
        n_components: int = 38,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        metrics: List[str] = ["mae", "rmse", "mape"],
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder_ckpt = encoder_ckpt
        self.tokenizer = load_tokenizer(tokenizer or encoder_ckpt)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        self.n_components = n_components

        # Load Encoder Model
        if Path(encoder_ckpt).exists():
            self.encoder = DeepSpeedMixin.load(encoder_ckpt).get_encoder()
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(
                encoder_ckpt,
                trust_remote_code=True,
            )

        # Validate the vocab size
        if vocab_size is not None:
            if hasattr(self.encoder, "config") and hasattr(
                self.encoder.config, "vocab_size"
            ):
                assert (
                    self.encoder.config.vocab_size == vocab_size
                ), f"Expected vocab size to match. got {self.encoder.config.vocab_size} and {vocab_size}"
        self.task_network = VFTDecayTaskHead(embed_dim=self.encoder.config.hidden_size)
        self.lossfn = torch.nn.MSELoss(reduction="mean")

        metrics = get_metrics(
            metrics,
            "regression",
            num_outputs=1,
            target_channels="ln k",
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, batch, return_all=False, **kwargs):  # type: ignore[override]
        mix_embedding = None
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state[:, 0, :]

            embedding = torch.stack(
                [
                    torch.mul(embedding[j, :], batch[f"composition_{i}"][j])
                    for j in range(embedding.shape[0])
                ]
            )
            if mix_embedding is None:
                mix_embedding = embedding
            else:
                mix_embedding += embedding

        params = self.task_network(mix_embedding, batch["temperature"])

        pred_unscaled = params["conductivity"]
        alpha = params["alpha"]
        beta = params["beta"]
        lmbda = params["beta"]

        exponent = torch.div(-1 * alpha + batch["composition_4"], lmbda)
        pred_decay = torch.mul((1 - beta), torch.exp(exponent)) + beta
        pred = torch.mul(pred_unscaled, pred_decay)

        # predicted conductivity*decay if salt molarity > alpha
        # else predicted conductivity
        pred = torch.where(batch["composition_4"] > alpha, pred, pred_unscaled)

        if return_all:
            return pred.view(-1, 1), params
        return pred.view(-1, 1), alpha

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds, alpha = self.forward(batch, return_all=False)
        target = batch["target"]
        loss = self.lossfn(preds, target) + alpha.abs().mean()
        return preds, loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.train_metrics.update(preds, batch["target"])
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            self.train_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_metrics.update(preds, batch["target"])
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.test_metrics.update(preds, batch["target"])
        return loss

    def on_test_epoch_end(self):
        self.log_dict(
            self.test_metrics.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.test_metrics.reset()

    def predict_step(self, batch, *args):
        mix_embedding = None
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)

            embedding = torch.stack(
                [
                    torch.mul(embedding[j, :], batch[f"composition_{i}"][j])
                    for j in range(embedding.shape[0])
                ]
            )
            if mix_embedding is None:
                mix_embedding = embedding
            else:
                mix_embedding += embedding

        preds = self.task_network(mix_embedding, batch["temperature"])

        out = {"embedding": embedding, "prediction": preds}

        for key in ["target", "is_oov"]:
            if key in batch.keys():
                out[key] = batch[key]

        return out

    def configure_optimizers(self):
        learnable_params = self.task_network.parameters()
        if not self.freeze_encoder:
            learnable_params = chain(learnable_params, self.encoder.parameters())

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
