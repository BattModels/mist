import logging
import importlib
import json
import os
from typing import Optional
from pathlib import Path

import torch
from jsonargparse import Namespace
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import WandbLogger


class SaveConfigWithCkpts(Callback):
    """Save Configuration with the model's checkpoints

    # Versions: Use Semantic Versioning for Model Checkpoint format

    ## Unnamed:
        - Stored `trainer.lightning_model.hparams` in model_hparms.json

    ## 0.2.0
        - Added version field to model_hparams.json
        - Added "class_path" field
        - Moved model hparams to "init_args" field

    ## 0.2.1
        - Save `JOB_CONFIG` to `job_config.json`

    ## 0.3.0
        - Save hyperparameters under `lighting_module` and `datamodule`.
    """

    VERSION = "0.3.0"

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        overwrite: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.overwrite = overwrite
        self.already_saved = False
        self.config_path = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        self.config_path = self.log_dir(trainer)

        if trainer.is_global_zero:
            self.config_path.mkdir(parents=True, exist_ok=True)
            config_json = self.parser.dump(
                self.config,
                skip_none=False,
                skip_check=True,
                skip_link_targets=False,
                format="json",
            )
            with open(Path(self.config_path, "config.json"), "w") as config_file:
                config_file.write(config_json)

            # Save full job config
            job_config = json.loads(os.environ.get("JOB_CONFIG", "{}"))
            with open(Path(self.config_path, "job_config.json"), "w") as fid:
                json.dump(job_config, fid)

            # Save model hyperparameters
            with open(Path(self.config_path, "model_hparams.json"), "w") as fid:
                model_cls = trainer.lightning_module.__class__
                model_config = {
                    "version": self.VERSION,
                    "class_path": f"{model_cls.__module__}.{model_cls.__name__}",
                    "lightning_module": trainer.lightning_module.hparams,
                    "datamodule": trainer.datamodule.hparams,
                }
                json.dump(model_config, fid, default=lambda x: str(type(x)))

            # Save Environment
            with open(Path(self.config_path, "env.json"), "w") as fid:
                json.dump(dict(os.environ), fid, sort_keys=True)

            if logger := trainer.logger:
                logger.log_hyperparams({"cli": self.config.as_dict()})

    @staticmethod
    def log_dir(trainer: Trainer) -> Path:
        log_dir = trainer.log_dir or trainer.default_root_dir
        logger = trainer.logger
        if logger is not None and isinstance(logger, WandbLogger):
            config_path = Path(log_dir, str(logger.name), str(logger.version))
        else:
            config_path = Path(log_dir)
        return config_path

    @staticmethod
    def instantiate(
        config_path: Path, max_position_embeddings: Optional[int] = None
    ) -> LightningModule:
        """Instantiate a model from a checkpoint but don't load weights"""
        with open(config_path, "r") as fid:
            config = json.load(fid)

        # Get model class name and config
        if version := config.get("version", None):
            if version.startswith("0.3"):
                cls_name, model_config = norm_class_config(
                    config["lightning_module"],
                    class_path=config.get("class_path", None),
                )
                model_config["vocab_size"] = config["datamodule"]["vocab_size"]

            else:
                cls_name, model_config = norm_class_config(config)
        else:
            cls_name = "electrolyte_fm.models.roberta_base.RoBERTa"
            model_config = config

        # Import the model class and initialize the model
        import_path = cls_name.split(".")
        if max_position_embeddings is not None:
            model_config["max_position_embeddings"] = max_position_embeddings
        model_cls = importlib.import_module(
            ".".join(import_path[:-1])
        ).__getattribute__(import_path[-1])
        assert import_path[-1] == model_cls.__name__
        model = model_cls(**model_config)

        if hasattr(model, "configure_model"):
            model.configure_model()

        return model

    @staticmethod
    def load(
        checkpoint_dir: str | Path,
        config_path=None,
        map_location=None,
        max_position_embeddings: Optional[int] = None,
    ) -> LightningModule:
        """Restore from a deepspeed checkpoint, mainly used for downstream tasks"""
        checkpoint_dir = Path(checkpoint_dir).resolve()
        config_path = config_path or checkpoint_dir.parent.parent.joinpath(
            "model_hparams.json"
        )
        assert (
            checkpoint_dir.exists()
        ), f"Missing deepspeed checkpoint directory: {checkpoint_dir}"
        assert config_path.is_file(), f"Missing model config file {config_path}"

        model = SaveConfigWithCkpts.instantiate(config_path, max_position_embeddings)

        # Fallback to cpu if no GPU
        if not torch.cuda.is_available() and map_location is None:
            map_location = torch.device("cpu")

        if checkpoint_dir.is_file():
            state = torch.load(
                checkpoint_dir, map_location=map_location, weights_only=False
            )
            if max_position_embeddings is not None:
                state = adjust_state_position_embeddings(state, max_position_embeddings)

            model.load_state_dict(state["state_dict"], strict=True, assign=True)
            return model

        # Load model weights from the checkpoint
        try:
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )

            state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
            if max_position_embeddings is not None:
                state = adjust_state_position_embeddings(state, max_position_embeddings)

            model.load_state_dict(state, strict=False, assign=True)
        except FileNotFoundError:
            logging.error(
                "failed to load checkpoint %s, trying to load rank 0 model states",
                checkpoint_dir,
            )
            file = Path(checkpoint_dir, "checkpoint", "mp_rank_00_model_states.pt")
            state = torch.load(file, map_location=map_location)
            logging.info("loaded %s", file)
            if max_position_embeddings is not None:
                state = adjust_state_position_embeddings(state, max_position_embeddings)

            model.load_state_dict(state["module"], strict=True, assign=True)

        return model


def adjust_state_position_embeddings(state, max_position_embeddings):
    module_state = state["module"]
    assert (
        "model.roberta_prelayernorm.embeddings.position_embeddings.weight"
        in module_state
    ), "Changing position embedding size only implemented for RoBERTaPreLayerNorm"
    current_max_pos, embed_size = module_state[
        "model.roberta_prelayernorm.embeddings.position_embeddings.weight"
    ].shape
    assert (
        max_position_embeddings > current_max_pos
    ), "Maximum position embedding cannot be decreased"
    # Initialize new position embedding matrix
    new_pos_embed = module_state[
        "model.roberta_prelayernorm.embeddings.position_embeddings.weight"
    ].new_empty(max_position_embeddings, embed_size)
    # Restore pre-train position embeddings
    new_pos_embed[:current_max_pos, :] = module_state[
        "model.roberta_prelayernorm.embeddings.position_embeddings.weight"
    ]
    module_state[
        "model.roberta_prelayernorm.embeddings.position_embeddings.weight"
    ].data = new_pos_embed
    state["module"] = module_state
    return state


def get_ckpt_tokenizer(path: str | Path) -> str:
    path = Path(path)
    config_path = path.parent.parent.joinpath("config.json")
    if not config_path.is_file():
        return str(path)
    with open(config_path, "r") as fid:
        config = json.load(fid)
    try:
        return config["data"]["tokenizer"]
    except KeyError:
        return config["data"]["init_args"]["tokenizer"]


def norm_class_config(config: dict, class_path: Optional[str] = None) -> (str, dict):
    """Parse a dictionary of hparams for a class name and init args"""
    init_args = dict()
    if "init_args" in config:
        init_args = config.pop("init_args")

    if "class_path" in config:
        class_path = config["class_path"]
    elif "_class_path" in config:
        class_path = config.pop("_class_path")
        init_args = config

    # Remove instantiator from init_args
    init_args.pop("_instantiator", None)
    init_args.pop("instantiator", None)

    return class_path, init_args
