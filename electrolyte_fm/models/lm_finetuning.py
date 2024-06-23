from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanAbsoluteError
from pathlib import Path
from .model_utils import DeepSpeedMixin
from .prediction_task_head import PredictionTaskHead
import yaml 

class LMFinetuning(pl.LightningModule, DeepSpeedMixin):
    """
    PyTorch Lightning module for finetuning LM encoder model on multiple tasks.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        task_config: Union[str, Path] = "",
        freeze_encoder: bool = False,
        learning_rate: float = 1.6e-4,
        dropout: float = 0.2,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoder_ckpt = encoder_ckpt
        self.task_config: Path = task_config 
        assert self.task_config.is_file()
        self.setup_task_config()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.encoder = DeepSpeedMixin.load(encoder_ckpt).get_encoder()
        self.save_hyperparameters(logger=False)

        head_hyperparams = {
            "embed_dim": self.encoder.config.hidden_size,
            "dropout": self.dropout,
        }

        self.task_networks = torch.nn.ModuleDict(
            {
                spec["measure_name"]: PredictionTaskHead(
                    **head_hyperparams, output_size=spec.get("n_classes", 1)
                )
                for spec in self.task_specs
            }
        )
        self.classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.regression_loss = torch.nn.MSELoss()
        self.setup_loss_functions()
        self.setup_metrics()
        self.freeze_encoder = freeze_encoder

    def setup_task_config(self) -> None:
        self.task_specs = yaml.safe_load(self.task_config.open())

    def setup_loss_functions(self):
        for spec in self.task_specs:
            if spec.get("n_classes", 1) > 1:
                spec["loss"] = self.classification_loss
            else:
                spec["loss"] = self.regression_loss

    def setup_metrics(self):
        for spec in self.task_specs:
            targets = spec.get("n_classes", 1)
            if targets == 1:  # regression problem
                spec["metric"] = MeanAbsoluteError().cuda()
            else:  # multi-class classification
                spec["metric"] = Accuracy(
                    task="multiclass", num_classes=targets, ignore_index=-1
                )

    def forward(self, batch, **kwargs):  # type: ignore[override]
        embedding = self.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
            return_dict=False,
        )

        sequence_output = embedding[0]
        out = {
            spec["measure_name"]: self.task_networks[spec["measure_name"]](
                sequence_output
            )
            for spec in self.task_specs
        }
        return out

    def batch_loss(self, outputs, batch):
        loss = 0
        for spec in self.task_specs:
            target = spec["measure_name"]
            w = spec.get("loss_weight", 1 / len(self.task_specs))
            if spec.get("n_classes", 1) > 1:
                spec_loss = w * spec["loss"](
                    outputs[target], batch[target].to(torch.int64)
                )
            else:
                labels = batch[target].reshape(batch[target].size()[0], 1)
                spec_loss = w * spec["loss"](outputs[target], labels)
            if not torch.isnan(spec_loss):
                loss += spec_loss
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = self.batch_loss(outputs, batch)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for spec in self.task_specs:
            target = spec["measure_name"]
            labels = batch[target]
            if spec.get("n_classes", 1) <= 1:
                labels = labels.reshape(labels.size()[0], 1)

            self.log(
                f"train/{target}_{spec['metric'].__class__.__name__}",
                spec["metric"].to(loss.device)(outputs[target], labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = self.batch_loss(outputs, batch)

        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        for spec in self.task_specs:
            target = spec["measure_name"]
            labels = batch[target]
            if spec.get("n_classes", 1) <= 1:
                labels = labels.reshape(labels.size()[0], 1)

            self.log(
                f"val/{target}_{spec['metric'].__class__.__name__}",
                spec["metric"].to(loss.device)(outputs[target], labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = self.batch_loss(outputs, batch)

        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for spec in self.task_specs:
            target = spec["measure_name"]
            labels = batch[target]
            if spec.get("n_classes", 1) <= 1:
                labels = labels.reshape(labels.size()[0], 1)
            self.log(
                f"test/{target}_{spec['metric'].__class__.__name__}",
                spec["metric"].to(loss.device)(outputs[target], labels),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def configure_optimizers(self):
        learnable_params = [
            p
            for _, task_network in self.task_networks.items()
            for p in task_network.parameters()
        ]
        if not self.freeze_encoder:
            learnable_params.extend([p for p in self.encoder.parameters()])

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
