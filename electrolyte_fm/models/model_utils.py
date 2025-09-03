import json
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection
from transformers import CONFIG_MAPPING as HF_CONFIG_MAPPING
from transformers import AutoConfig, PretrainedConfig

from ..utils.ckpt import SaveConfigWithCkpts


def load_encoder(name_or_path: str):
    if Path(name_or_path).exists():
        return DeepSpeedMixin.load(name_or_path).get_encoder()
    else:
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            name_or_path,
            trust_remote_code=True,
        )


class DeepSpeedMixin:
    @staticmethod
    def load(checkpoint_dir, **kwargs):
        print(checkpoint_dir)
        return SaveConfigWithCkpts.load(checkpoint_dir, **kwargs)

    def load_state(self, checkpoint_dir):
        print("Loading state for checkpoint:", checkpoint_dir)
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )

        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        self.load_state_dict(state, strict=False, assign=True)

    def get_encoder(self):
        raise NotImplementedError


class LoggingMixin:
    def on_train_epoch_start(self) -> None:
        # Update the dataset's internal epoch counter

        self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)
        self.log(
            "train/dataloader_epoch",
            self.trainer.train_dataloader.dataset._epoch,
            rank_zero_only=True,
            sync_dist=True,
        )
        return super().on_train_epoch_start()


def record_summary_stats(logger, metrics: MetricCollection):
    if isinstance(logger, WandbLogger):
        define_metric = logger.experiment.define_metric
        for name, metric in metrics.items():
            if (
                not hasattr(metric, "higher_is_better")
                or metric.higher_is_better is None
            ):
                continue
            for phase in ["", "_epoch", "_step"]:
                define_metric(
                    name + phase,
                    summary="last,best",
                    goal="maximize" if metric.higher_is_better else "minimize",
                )


def record_loss_summary_stats(logger):
    if isinstance(logger, WandbLogger):
        define_metric = logger.experiment.define_metric
        for m in ["train/loss", "val/loss", "test/loss"]:
            for s in ["", "_step", "_epoch"]:
                define_metric(m + s, summary="last,best,min", goal="minimize")


class CanSkip:
    def should_skip(self):
        """Return true if the model should skip this batch, the model
        should still perform a forward pass, but return `0 * loss`
        instead. Do not return `None` as unsupported by DeepSpeed
        """
        if hasattr(self, "skip_this_batch") and self.skip_this_batch:
            return True
        else:
            return False


@dataclass
class ModelConfig:
    encoder: PretrainedConfig

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "encoder"}
        d["encoder"] = self.encoder.to_diff_dict()
        d["encoder"]["model_type"] = self.encoder.model_type
        if hasattr(self.encoder, "_name_or_path") and self.encoder._name_or_path:
            d["encoder"]["_name_or_path"] = self.encoder._name_or_path
        return d

    @classmethod
    def from_dict(cls, d: dict):
        encoder_config = d.pop("encoder")
        if (
            encoder_config["model_type"] not in HF_CONFIG_MAPPING
            and "_name_or_path" in encoder_config
        ):
            model_id = encoder_config.pop("_name_or_path")
            encoder = AutoConfig.from_pretrained(
                model_id, **encoder_config, trust_remote_code=True
            )
        else:
            encoder = AutoConfig.for_model(**encoder_config)
        return cls(encoder, **d)

    def to_json_file(self, config_file: str | Path):
        Path(config_file).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json_file(cls, config_file: str | Path):
        return cls.from_dict(json.loads(Path(config_file).read_text()))
