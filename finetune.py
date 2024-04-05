import os
import torch

from datetime import timedelta
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import (
    LightningCLI,
    LightningArgumentParser,
)
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from electrolyte_fm.utils.callbacks import ThroughputMonitor
from jsonargparse import lazy_instance

# classes passed via cli
from electrolyte_fm.models.lm_classification import LMClassification
from electrolyte_fm.models.lm_regression import LMRegression
from electrolyte_fm.models.property_prediction_dataset import PropertyPredictionDataModule
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts

class MyLightningCLI(LightningCLI):
    def before_fit(self):
        self.trainer.logger.log_hyperparams(
            {
                "n_gpus_per_node": self.trainer.num_devices,
                "n_nodes": self.trainer.num_nodes,
                "world_size": self.trainer.world_size,
            }
        )


def cli_main(args=None):
    monitor = "val/loss_epoch"
    callbacks = [
        ThroughputMonitor(),
        EarlyStopping(monitor=monitor),
        ModelCheckpoint(
            save_last="link",
            filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.2f}",
            monitor=monitor,
            save_top_k=5,
            verbose=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            save_top_k=2,
            monitor="step",
            verbose=True,
            mode="max",
            train_time_interval=timedelta(minutes=30),
            auto_insert_metric_name=False,
        ),
    ]

    num_nodes = int(os.environ.get("NRANKS", 1))
    rank = int(os.environ.get("GLOBAL_RANK", 0))
    os.environ["NODE_RANK"] = str(rank % num_nodes)
    os.environ["GLOBAL_RANK"] = str(rank % num_nodes)

    print(f"PY: NUM_NODES: {num_nodes} PMI_RANK: {rank} PID {os.getpid()}")

    if rank is not None and int(rank) != 0:
        logger = None
    else:
        logger = lazy_instance(WandbLogger, project="mist", save_code=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    return MyLightningCLI(
        datamodule_class=PropertyPredictionDataModule,
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "precision": "16-mixed",
            "devices": -1,
            "num_nodes": num_nodes or 1,
            "strategy": "deepspeed",
            "use_distributed_sampler": True,  # Handled by DataModule (Needed as Iterable)
            "profiler": {
                "class_path": "pytorch_lightning.profilers.PyTorchProfiler",
                "init_args": {
                    "emit_nvtx": True,
                },
            },
        },
        save_config_callback=SaveConfigWithCkpts,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=args is None,  # support unit testing
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()