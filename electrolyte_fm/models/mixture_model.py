import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanAbsoluteError

from .model_utils import DeepSpeedMixin, LoggingMixin
from .normalize import Standardize
from .prediction_task_head import PredictionTaskHead


class MixtureModel(LightningModule, DeepSpeedMixin, LoggingMixin):
    """
    PyTorch Lightning module for mixture property prediction.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        output_size: int = 1,
        dropout: float = 0.1,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_schedule: LRSchedulerCallable | None = None,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.save_hyperparameters(ignore=["optimizer", "lr_schedule"])
        self.task_network = PredictionTaskHead(
            embed_dim=hidden_size,
            output_size=output_size,
            dropout=dropout,
        )
        self.lossfn = torch.nn.MSELoss(reduction="mean")
        self.metric = MeanAbsoluteError()
        self.transform = Standardize(output_size)

    def on_fit_start(self):
        """Standardized training data"""
        if self.global_rank == 0:
            assert self.trainer.datamodule.target_dataset is not None
            ds = self.trainer.datamodule.target_dataset
            state = self.transform.fit(ds)

        state = self.trainer.strategy.broadcast(state)
        self.transform.load_state_dict(state)

    def forward(self, batch, transform=True, **kwargs):  # type: ignore[override]
        pred_unscaled = self.task_network(batch["embedding"])
        if transform:
            return self.transform.forward(pred_unscaled)
        return pred_unscaled

    def setup(self, stage: str) -> None:
        if isinstance(self.logger, WandbLogger):
            for m in ["train/loss", "val/loss"]:
                for s in ["_step", "_epoch"]:
                    self.logger.experiment.define_metric(m + s, summary="min")

    def _scaled_pred_loss(self, batch):
        """Compute loss before transforming the model's predictions"""
        preds = self.forward(batch, transform=False)
        target = batch["target"]
        target = self.transform.inverse(target)
        loss = self.lossfn(preds, target)
        preds = self.transform.forward(preds)
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
        self.metric.update(preds, batch["target"])
        self.log(
            "train/mae",
            value=self.metric.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.metric.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        self.metric.update(preds, batch["target"])
        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val/mae",
            value=self.metric.compute(),
            on_epoch=True,
            sync_dist=True,
        )
        self.metric.reset()

    def test_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        preds, loss = self._scaled_pred_loss(batch)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("test/mae", self.metric)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if schedule := self.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": schedule(optimizer), "interval": "step"},
            }
        return optimizer
