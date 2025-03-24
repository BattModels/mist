import logging
from itertools import chain
from typing import Any, Callable

import lightning.pytorch as pl
import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from electrolyte_fm.models.normalize import AbstractNormalizer, IdentityTransform

TaskNetworkCallable = Callable[Any, nn.Module]


def masked_mse_loss(
    preds: torch.Tensor, target: torch.Tensor, mask: torch.BoolTensor
) -> torch.Tensor:
    loss = F.mse_loss(preds, target, reduction="none") * mask
    return loss.sum() / mask.sum()


def distance_matrix_loss(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    loss = []
    for bdx in range(y_hat.shape[0]):
        pos_mask = mask[bdx]
        pos_ref = y[bdx, pos_mask, :]
        pos_hat = y_hat[bdx, pos_mask, :]
        dist_hat = torch.cdist(pos_hat, pos_hat)
        dist_expt = torch.cdist(pos_ref, pos_ref)
        loss.append(F.mse_loss(dist_hat, dist_expt))

    return torch.tensor(loss).mean()


class TokenLevelPredictor(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        seq_network: nn.Module,
        token_network: nn.Module,
        seq_transform: AbstractNormalizer | None = None,
        token_transform: AbstractNormalizer | None = None,
        distance_matrix_loss: bool = False,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        freeze_encoder: bool | str = False,
        lr_schedule: LRSchedulerCallable | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.seq_network = seq_network
        self.token_network = token_network
        self.seq_transform = seq_transform or IdentityTransform()
        self.token_transform = token_transform or IdentityTransform()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.freeze_encoder = freeze_encoder
        self.distance_matrix_loss = distance_matrix_loss
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def on_fit_start(self):
        """Standardized training data"""
        assert hasattr(self.trainer, "datamodule")
        ds = self.trainer.datamodule.train_dataset.take(1_000)

        # Sequence Transform
        ds_seq = ds.select_columns(["target", "target_mask"])
        self.seq_transform.leader_fit(
            ds_seq, self.global_rank, self.trainer.strategy.broadcast
        )

        # Token Transform
        ds_token = ds.select_columns(["token_target", "token_target_mask"])
        ds_token = ds_token.rename_columns(
            {"token_target": "target", "token_target_mask": "target_mask"}
        )
        self.token_transform.leader_fit(
            ds_token, self.global_rank, self.trainer.strategy.broadcast
        )
        logging.info(
            {
                "seq_transform": self.seq_transform.state_dict(),
                "token_transform": self.token_transform.state_dict(),
            }
        )

    def forward(self, input_ids, attention_mask=None):
        hs = self.encoder(input_ids, attention_mask).last_hidden_state
        y_mol = self.seq_transform.forward(self.seq_network(hs))
        y_token = self.token_transform.forward(self.token_network(hs))
        return y_mol, y_token

    def forward_with_loss(self, batch):
        hs = self.encoder(
            batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        y_mol_raw = self.seq_network(hs)
        y_hat_token = self.token_network(hs)

        y_token_ref = self.token_transform.inverse(batch["token_target"])
        y_token_mask = batch["token_target_mask"]

        loss_seq = masked_mse_loss(
            y_mol_raw,
            self.seq_transform.inverse(batch["target"]),
            batch["target_mask"],
        )

        # Treat the final 3 channels as atomic coordinates
        if self.distance_matrix_loss:
            y_pos = y_hat_token[:, :, -3:]
            y_pos_ref = y_token_ref[:, :, -3:]
            y_pos_mask = y_token_mask[:, :, -3:].all(-1)
            loss_dist = distance_matrix_loss(y_pos, y_pos_ref, y_pos_mask)

            y_token = y_hat_token[:, :, :-3]
            y_token_mask = y_token_mask[:, :, :-3]
            y_token_ref = y_token_ref[:, :, :-3]

        else:
            y_token = y_hat_token
            loss_dist = None

        loss_token = masked_mse_loss(y_token, y_token_ref, y_token_mask)
        loss = loss_seq + loss_token + (loss_dist if loss_dist is not None else 0)

        out = {
            "loss": loss,
            "sequence": self.seq_transform.forward(y_mol_raw),
            "token": self.token_transform.forward(y_hat_token),
            "token_loss": loss_token,
            "seq_loss": loss_seq,
            "dist_loss": loss_dist,
        }

        return out

    def training_step(self, batch):
        out = self.forward_with_loss(batch)
        self.log_dict(
            {
                "train/loss": out["loss"],
                "train/loss_token": out["token_loss"],
                "train/seq_loss": out["seq_loss"],
                "train/dist_loss": out["dist_loss"],
            },
            on_step=True,
            on_epoch=True,
        )
        return out

    def validation_step(self, batch):
        out = self.forward_with_loss(batch)
        self.log_dict(
            {
                "val/loss": out["loss"],
                "val/loss_token": out["token_loss"],
                "val/seq_loss": out["seq_loss"],
                "val/dist_loss": out["dist_loss"],
            },
            on_step=False,
            on_epoch=True,
        )
        return out

    def configure_optimizers(self):
        learnable_params = chain(
            self.seq_network.parameters(), self.token_network.parameters()
        )
        if not self.freeze_encoder:
            learnable_params = chain(learnable_params, self.encoder.parameters())
        elif self.freeze_encoder == "encoder":
            learnable_params = chain(
                learnable_params, self.encoder.embeddings.parameters()
            )

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer


if __name__ == "__main__":
    from datetime import timedelta
    from os import environ

    from jsonargparse import lazy_instance
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from electrolyte_fm.utils.callbacks import ThroughputMonitor
    from electrolyte_fm.utils.cli import MistLightningCLI

    logging.basicConfig(level=logging.INFO)
    monitor = "val/loss_epoch"
    val_loss_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.3f}",
        monitor=monitor,
        save_top_k=2,
        verbose=True,
        save_last="link",
        enable_version_counter=True,
        auto_insert_metric_name=False,
        save_weights_only=True,
    )
    val_loss_ckpt.CHECKPOINT_NAME_LAST = "best"
    step_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}",
        monitor="step",
        verbose=True,
        mode="max",
        save_top_k=2,
        save_last=True,
        train_time_interval=timedelta(minutes=15),
        auto_insert_metric_name=False,
    )
    step_ckpt.CHECKPOINT_NAME_LAST = "last"

    torch.set_float32_matmul_precision("high")
    cli = MistLightningCLI(
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "enable_progress_bar": False,
            "callbacks": [
                ThroughputMonitor(),
                LearningRateMonitor("step"),
                val_loss_ckpt,
                step_ckpt,
            ],
            "logger": lazy_instance(
                WandbLogger,
                project="partial-charge",
                save_code=True,
                id=environ.get("WANDB_ID", None),
                resume=environ.get("WANBD_RESUME", "allow"),
            ),
            "max_epochs": 1000,
            "precision": "16-mixed",
        },
        parser_kwargs={"parser_mode": "jsonnet"},
        run=False,
    )

    trainer: pl.Trainer = cli.trainer
    model: TokenLevelPredictor = cli.model
    trainer.fit(model, cli.datamodule)
