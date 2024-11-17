import json
import os
import logging
from datetime import timedelta

import _jsonnet as jsonnet  # Unused, but otherwise we get glibc errors on delta 🫠
import torch
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI,
    _InstantiatorFn,
    _get_module_type,
)
from pytorch_lightning.loggers import WandbLogger
from electrolyte_fm.utils.callbacks import ThroughputMonitor, SpikeDetection
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        if self.trainer.logger is not None and hasattr(
            self.trainer.logger, "log_hyperparams"
        ):
            job_config = json.loads(os.environ.get("JOB_CONFIG", "{}"))
            self.trainer.logger.log_hyperparams(
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

        # WARN: electrolyte_fm.utils.SaveConfigWithCkpts with any new linked arguments
        # prefer apply_on parse over instantiate

        # Set model vocab_size from the dataset's vocab size
        parser.link_arguments(
            "data.vocab_size", "model.init_args.vocab_size", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.target_columns",
            "model.init_args.target_columns",
            apply_on="instantiate",
        )

    def _add_instantiators(self) -> None:
        self.config_dump = json.loads(
            self.parser.dump(
                self.config, skip_link_targets=False, skip_none=False, format="json"
            )
        )
        if "subcommand" in self.config:
            self.config_dump = self.config_dump[self.config.subcommand]

        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="model"),
            _get_module_type(self._model_class),
            subclasses=self.subclass_mode_model,
        )
        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="data"),
            _get_module_type(self._datamodule_class),
            subclasses=self.subclass_mode_data,
        )


def cli_main(args=None):
    monitor = "val/loss_epoch"
    callbacks = [
        ThroughputMonitor(),
        SpikeDetection(atol=0.3, warmup=800, finite_only=False),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.2f}",
            monitor=monitor,
            save_top_k=2,
            verbose=True,
            save_last="link",
            enable_version_counter=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            monitor="step",
            verbose=True,
            mode="max",
            save_top_k=2,
            save_last=False,
            train_time_interval=timedelta(minutes=20),
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
            project="mist",
            save_code=True,
            id=os.environ.get("WANDB_ID", None),
            resume=os.environ.get("WANBD_RESUME", "allow"),
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
        seed_everything_default=42,
        parser_kwargs={"parser_mode": "jsonnet", "default_env": True},
        run=args is None,  # support unit testing
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(name)s} %(levelname)s - %(message)s",
    )
    cli_main()
