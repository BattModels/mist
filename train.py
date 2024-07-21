import json
import os
from datetime import timedelta

import torch
from jsonargparse import lazy_instance
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.loggers import WandbLogger

from electrolyte_fm.data_modules import PropertyPredictionDataModule, RobertaDataSet
from electrolyte_fm.models.lm_finetuning import LMFinetuning

# classes passed via cli
from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.utils.callbacks import SpikeDetection, ThroughputMonitor
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts


class MyLightningCLI(LightningCLI):

    def before_fit(self):
        if logger := self.trainer.logger:
            job_config = json.loads(os.environ.get("JOB_CONFIG", "{}"))

            logger.log_hyperparams(
                {
                    "job_config": job_config,
                    "n_gpus_per_node": self.trainer.num_devices,
                    "n_nodes": self.trainer.num_nodes,
                    "world_size": self.trainer.world_size,
                }
            )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--tags",
            type=list,
            help="Tags for WandB logger",
            default=[],
        )
        parser.link_arguments("tags", "trainer.logger.init_args.tags")

        # Set model vocab_size from the dataset's vocab size
        parser.link_arguments(
            "data.vocab_size", "model.init_args.vocab_size", apply_on="instantiate"
        )
        # Set model task_specs from the dataset's task_specs
        parser.link_arguments(
            "data.task_specs", "model.init_args.task_specs", apply_on="instantiate"
        )

        # Configure tokenizer from checkpoint
        parser.link_arguments(
            "model.init_args.encoder_ckpt",
            "data.init_args.tokenizer",
            compute_fn=SaveConfigWithCkpts.get_ckpt_tokenizer,
        )


def cli_main(args=None):
    monitor = "val/loss_epoch"
    callbacks = [
        ThroughputMonitor(),
        SpikeDetection(
            atol=0.3,
            warmup=200, 
            finite_only=False
            ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.2f}",
            monitor=monitor,
            save_last="link",
            save_top_k=2,
            verbose=True,
            enable_version_counter=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            monitor="step",
            verbose=True,
            save_last=False,
            mode="max",
            save_top_k=5,
            train_time_interval=timedelta(minutes=15),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor("step"),
    ]

    rank = int(os.environ.get("MIST_PID_RANK", 0))
    if rank is not None and int(rank) != 0:
        logger = None
    else:
        logger = lazy_instance(
            WandbLogger,
            id=os.environ.get("SLURM_JOB_ID", "test"),
            project="mist",
            save_code=True,
        )

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    return MyLightningCLI(
        trainer_defaults={
            "callbacks": callbacks,
            "logger": logger,
            "precision": "16-mixed",
            "strategy": "deepspeed",
            "use_distributed_sampler": False,  # Handled by DataModule (Needed as Iterable)
        },
        save_config_callback=SaveConfigWithCkpts,
        save_config_kwargs={"overwrite": True},
        args=args,
        subclass_mode_model=False,
        run=args is None,  # support unit testing
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cli_main()
