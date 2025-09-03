import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from transformers import AutoModel, AutoConfig, PretrainedConfig
from transformers import CONFIG_MAPPING as HF_CONFIG_MAPPING
from jsonargparse import class_from_function

from torchmetrics import MetricCollection
from ..utils.metrics import (
    setup_channel_metrics,
    masked_loss,
    masked_metric_regression_update,
    bootstrap_collection,
)
from .model_utils import ModelConfig
from .normalize import AbstractNormalizer
from .prediction_task_head import PredictionTaskHead


@dataclass
class SparseRegressionConfig(ModelConfig):
    target_columns: list[str]
    transform: str | list[str] = "standardize"
    dropout: float = 0.1

    @property
    def num_targets(self):
        return len(self.target_columns) if self.target_columns is not None else 1


class SparseRegressionModel(nn.Module):
    def __init__(self, config: SparseRegressionConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_config(config.encoder, add_pooling_layer=False)
        self.transform = AbstractNormalizer.get(config.transform, config.num_targets)
        self.task_network = PredictionTaskHead(
            self.config.encoder.hidden_size,
            self.config.num_targets,
            self.config.dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        transform: bool = True,
    ):
        hs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
        ).last_hidden_state
        y = self.task_network(hs)
        if transform:
            return self.transform.forward(y)
        return y

    def save_pretrained(self, save_directory: str | Path):
        from safetensors.torch import save_model

        save_directory = Path(save_directory)
        save_directory.mkdir(exist_ok=True, parents=True)
        save_model(self, str(save_directory.joinpath("model.safetensors")))
        self.config.to_json_file(save_directory.joinpath("config.json"))

    @classmethod
    def from_pretrained(cls, save_directory: str | Path):
        from safetensors.torch import load_model

        save_directory = Path(save_directory)
        model = cls(
            SparseRegressionConfig.from_json_file(
                save_directory.joinpath("config.json")
            )
        )
        load_model(model, save_directory.joinpath("model.safetensors"))
        return model

    @classmethod
    def from_pretrained_encoder(
        cls,
        name_or_path,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> Self:
        encoder = AutoModel.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            add_pooling_layer=False,
        )
        config = SparseRegressionConfig(encoder=encoder.config, **kwargs)
        model = cls(config)
        model.encoder = encoder
        return model


SparseRegressionFromPretrainedEncoder = class_from_function(
    func=SparseRegressionModel.from_pretrained_encoder,
    func_return=SparseRegressionModel,
    name="SparseRegressionFromPretrainedEncoder",
)


class SparseRegressionLightningModel(LightningModule):
    """
    PyTorch Lightning module for training on sparse regression data
    """

    def __init__(
        self,
        model: SparseRegressionModel | SparseRegressionConfig,
        encoder_ckpt: Path | str | None = None,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
        sparsity_weighted_loss: bool = False,
        metrics: list[str] = ["rmse", "mae", "r2"],
        bootstrap: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = (
            model
            if isinstance(model, SparseRegressionModel)
            else SparseRegressionModel(model)
        )
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lossfn = nn.MSELoss(reduction="none")
        self.target_metrics = metrics

        # Setup metrics
        self.target_metrics = metrics
        mc, clean_target_names = setup_channel_metrics(
            self.model.config.target_columns, metrics
        )
        self.clean_target_names = clean_target_names
        self.train_metrics = mc.clone(prefix="train/")

        if bootstrap:
            mc = bootstrap_collection(mc, num_bootstraps=50)

        self.val_metrics = mc.clone(prefix="val/")
        self.test_metrics = mc.clone(prefix="test/")

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["", "_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min,last")

    def on_fit_start(self):
        """Standardized training data"""
        self.model.transform.leader_fit(
            self.trainer.datamodule.target_dataset,
            self.global_rank,
            self.trainer.strategy.broadcast,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_loss(self, **kwargs):
        pred_unscaled = self.forward(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            transform=False,
        )
        loss = masked_loss(
            self.lossfn,
            pred_unscaled,
            self.model.transform.inverse(kwargs["target"]),
            kwargs["target_mask"],
        )
        preds = self.model.transform.forward(pred_unscaled)
        return preds, loss

    def _masked_metric(self, metrics: MetricCollection, y, y_ref, mask, targets):
        for idx, target in enumerate(targets):
            y_pred = y[:, idx]
            y_true = y_ref[:, idx]
            target_mask = mask[:, idx]

            if not target_mask.any():
                continue

            y_pred = y_pred[target_mask]
            y_true = y_true[target_mask]
            for metric in self.target_metrics:
                metrics[f"{target}/{metric}"].update(y_pred, y_true)

    def stage_step(self, stage: str, batch):
        preds, loss = self.forward_loss(**batch)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        metrics = getattr(self, f"{stage}_metrics")
        masked_metric_regression_update(
            metrics,
            preds,
            batch["target"],
            batch["target_mask"],
            target_names=self.clean_target_names,
            metric_names=self.target_metrics,
        )

        self.log_dict(metrics.compute(), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("train", batch)

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("val", batch)

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        return self.stage_step("test", batch)

    def configure_optimizers(self):
        learnable_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
                continue
            learnable_params.append(param)

        optimizer = self.optimizer(learnable_params)
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
