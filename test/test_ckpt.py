import json
from unittest import mock
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.demos.boring_classes import BoringModel, BoringDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from train import cli_main
from electrolyte_fm.utils.tokenizer import load_tokenizer
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts, get_ckpt_tokenizer


class MockedModel(BoringModel):
    def __init__(self, vocab_size: int):
        self.save_hyperparameters()
        super().__init__()


class MockedData(BoringDataModule):
    def __init__(self, tokenizer: str, batch_size: int = 32):
        super().__init__()
        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.batch_size = batch_size
        self.val_batch_size = batch_size
        # save_hyperparameters only saves args to the model
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters()


@pytest.fixture()
def cli(tmp_path):
    cli = cli_main(
        [
            "--model=test.test_ckpt.MockedModel",
            "--data=test.test_ckpt.MockedData",
            "--data.tokenizer=smirk",
            "--trainer.strategy=ddp",
            "--trainer.max_steps=5",
            "--trainer.enable_progress_bar=false",
            "--trainer.enable_model_summary=false",
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.logger=WandbLogger",
            f"--trainer.logger.init_args.save_dir={tmp_path}",  # needed to redirect wandb
        ]
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    return cli


def get_single_callback(cls, callbacks):
    cb = list(filter(lambda cb: cb.__class__.__name__ == cls.__name__, callbacks))
    assert len(cb) == 1
    return cb[0]


def test_ckpt(cli):
    # Locate callback
    cb = get_single_callback(SaveConfigWithCkpts, cli.trainer.callbacks)
    assert cb.config_path is not None
    assert cb.config_path.is_dir()
    assert cb.config_path.joinpath("config.json").is_file()
    assert cb.config_path.joinpath("model_hparams.json").is_file()

    # Check that the dataloader config is saved
    data_config = {
        "class_path": "test.test_ckpt.MockedData",
        "init_args": {"tokenizer": "smirk", "batch_size": 32},
    }
    with open(cb.config_path.joinpath("config.json"), "r") as fid:
        assert json.load(fid)["data"] == data_config

    # Check that the model config is saved
    with open(cb.config_path.joinpath("model_hparams.json"), "r") as fid:
        model_config = json.load(fid)
    assert model_config["class_path"] == __name__ + ".MockedModel"
    assert model_config["lightning_module"] == {
        "class_path": __name__ + ".MockedModel",
        "_instantiator": "pytorch_lightning.cli.instantiate_module",
        "init_args": {"vocab_size": None},
    }
    assert model_config["datamodule"] == {
        "_instantiator": "pytorch_lightning.cli.instantiate_module",
        "vocab_size": cli.datamodule.vocab_size,
        **data_config,
    }
    assert "version" in model_config.keys()
    assert model_config["version"] == "0.3.0"


def test_ckpt_load(cli):
    trainer = cli.trainer

    cb = get_single_callback(SaveConfigWithCkpts, trainer.callbacks)
    assert cb.config_path is not None
    assert cb.config_path.is_dir()
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    assert Path(trainer.checkpoint_callback.last_model_path).exists()

    # validate model_hparams
    config_path = cb.config_path.joinpath("model_hparams.json")
    assert config_path.is_file()
    model = SaveConfigWithCkpts.instantiate(config_path)
    assert isinstance(model, MockedModel)


def test_ckpt_tokenizer(cli):
    trainer = cli.trainer
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    ckpt = Path(trainer.checkpoint_callback.last_model_path)
    assert ckpt.exists()

    # Check that get_ckpt_tokenizer returns a tokenizer name
    tokenizer_name = get_ckpt_tokenizer(ckpt)
    assert tokenizer_name is not None and isinstance(tokenizer_name, str)

    # Validate tokenizer is viable
    tokenizer = load_tokenizer(tokenizer_name)
    assert tokenizer is not None and isinstance(tokenizer, PreTrainedTokenizerBase)
