# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import _jsonnet as jsonnet  # noqa: F401 Unused, but otherwise we get glibc errors on delta 🫠
import torch
from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import WandbLogger

from electrolyte_fm.utils.callbacks import SpikeDetection, ThroughputMonitor
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts
from electrolyte_fm.utils.cli import MistLightningCLI


class MyLightningCLI(MistLightningCLI):
    def _parse_ckpt_path(self) -> None:
        """Preserve the saved model class when resuming, including on Lightning 2.5."""
        subcommand = self.config.get("subcommand")
        if not subcommand:
            return
        if self.config[subcommand].get("resume_from_config", False):
            return
        checkpoint = self.config[subcommand].get("ckpt_path")
        if not checkpoint or not Path(checkpoint).is_file():
            return
        hparams = torch.load(checkpoint, weights_only=True, map_location="cpu").get(
            "hyper_parameters", {}
        )
        hparams.pop("_instantiator", None)
        if not hparams:
            return
        # Lightning's saved CLI parameters are flat with an internal class key;
        # jsonargparse needs the class_path/dict_kwargs representation instead.
        if "_class_path" in hparams:
            hparams = {
                "class_path": hparams.pop("_class_path"),
                "dict_kwargs": hparams,
            }
        self.config = self.parser.parse_object(
            {subcommand: {"model": hparams}}, self.config
        )

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
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--default_callbacks",
            type=bool,
            default=True,
            help="Include the entrypoint's default checkpoint and monitoring callbacks",
        )
        parser.add_argument(
            "--resume_from_config",
            type=bool,
            default=False,
            help="Construct from the supplied config while restoring checkpoint training state",
        )

        # WARN: electrolyte_fm.utils.SaveConfigWithCkpts with any new linked arguments
        # prefer apply_on parse over instantiate

        # Set model vocab_size from the dataset's vocab size
        parser.link_arguments(
            "data.vocab_size", "model.init_args.vocab_size", apply_on="instantiate"
        )

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config = self.config[self.subcommand] if self.subcommand else self.config
        if not config.default_callbacks:
            self.trainer_defaults["callbacks"] = []


def cli_main(args=None):
    monitor = "val/loss_epoch"
    val_loss_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.3f}",
        monitor=monitor,
        save_top_k=1,
        verbose=True,
        save_last="link",
        enable_version_counter=True,
        auto_insert_metric_name=False,
    )
    val_loss_ckpt.CHECKPOINT_NAME_LAST = "best"
    step_ckpt = ModelCheckpoint(
        filename="epoch={epoch}-step={step}",
        monitor="step",
        verbose=True,
        mode="max",
        save_top_k=1,
        save_last=True,
        train_time_interval=timedelta(minutes=15),
        auto_insert_metric_name=False,
    )
    step_ckpt.CHECKPOINT_NAME_LAST = "last"
    callbacks = [
        ThroughputMonitor(),
        # SpikeDetection(atol=0.3, warmup=800, finite_only=False),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_loss={" + monitor + ":.2f}",
            monitor=monitor,
            save_top_k=1,
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
            save_top_k=1,
            save_last=False,
            train_time_interval=timedelta(minutes=30),
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
            # Handled by DataModule (Needed as Iterable)
            "use_distributed_sampler": False,
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
