import json
from unittest import mock

import pytest
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from torch.utils.data import DataLoader

from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts


class MockedModel(LightningModule):
    def __init__(self, vocab_size: int, linked: int):
        self.save_hyperparameters()
        super().__init__()

    def forward(self, input):
        return input

    def training_step(self, batch, batch_idx):
        self.forward(batch)  # Mock calling forward
        return torch.zeros(1, requires_grad=True)

    def configure_optimizers(self):
        pass


class MockedData(LightningDataModule):
    def __init__(self, tokenizer: str, linked: int):
        self.tokenizer = tokenizer
        self.save_hyperparameters()
        super().__init__()

    def train_dataloader(self):
        return DataLoader(range(5), batch_size=1)


@pytest.fixture()
def cli(tmp_path):
    with mock.patch(
        "sys.argv",
        [
            "any.py",
            "--data.tokenizer=smirk",
            "--data.linked=10",
            "--model.vocab_size=256",
        ],
    ):
        parser = LightningArgumentParser()
        parser.add_class_arguments(MockedModel, "model")
        parser.add_class_arguments(MockedData, "data")
        parser.link_arguments("data.linked", "model.linked", apply_on="parse")
        parsed_args = dict(parser.parse_args())
        args_ = [
            "fit",
        ]
        args_.extend(["--" + k + "=" + str(v) for k, v in parsed_args.items()])

    _cli = LightningCLI(
        trainer_defaults={
            "max_steps": 2,
            "default_root_dir": tmp_path,
        },
        model_class=MockedModel,
        datamodule_class=MockedData,
        save_config_callback=SaveConfigWithCkpts,
        args=args_,
    )
    return _cli


def test_ckpt(cli):
    # Locate callback
    cb = list(
        filter(lambda cb: isinstance(cb, SaveConfigWithCkpts), cli.trainer.callbacks)
    )
    assert len(cb) == 1
    cb: SaveConfigWithCkpts = cb[0]

    assert cb.config_path is not None
    assert cb.config_path.is_dir()
    assert cb.config_path.joinpath("config.json").is_file()
    assert cb.config_path.joinpath("model_hparams.json").is_file()

    # Check that the dataloader config is saved
    data_config = {"linked": 10, "tokenizer": "smirk"}
    assert dict(cb.config["data"]) == data_config
    with open(cb.config_path.joinpath("config.json"), "r") as fid:
        assert json.load(fid)["data"] == data_config

    # Check that the model config is saved
    with open(cb.config_path.joinpath("model_hparams.json"), "r") as fid:
        model_config = json.load(fid)
    assert model_config["class_path"] == __name__ + ".MockedModel"
    assert model_config["lightning_module"] == {
        "_instantiator": "pytorch_lightning.cli.instantiate_module",
        "linked": 10,
        "vocab_size": 256,
    }

    assert model_config["datamodule"] == {
        "_instantiator": "pytorch_lightning.cli.instantiate_module",
        "linked": 10,
        "tokenizer": "smirk",
    }
    assert "version" in model_config.keys()
