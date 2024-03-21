import os
import torch
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
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.roberta_dataset import RobertaDataSet
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

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Set model vocab_size from the dataset's vocab size
        parser.link_arguments(
            "data.vocab_size", "model.vocab_size", apply_on="instantiate"
        )


def cli_main(args=None):
    monitor = "val/perplexity"
    callbacks = [
        ThroughputMonitor(),
        EarlyStopping(monitor=monitor),
        ModelCheckpoint(
            save_last="link",
            monitor=monitor,
            save_top_k=5,
            verbose=True,
        ),
    ]

    num_nodes = int(os.environ.get("NRANKS", 1))
    rank = int(os.environ.get("PMI_RANK", 1))
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
        model_class=RoBERTa,
        datamodule_class=RobertaDataSet,
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "precision": "16-mixed",
            "devices": -1,
            "num_nodes": num_nodes or 1,
            "strategy": "deepspeed",
            "use_distributed_sampler": False,  # Handled by DataModule (Needed as Iterable)
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
