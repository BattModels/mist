import pytest
import json
from pathlib import Path
from random import randint
from tempfile import TemporaryDirectory

from deepspeed.utils import zero_to_fp32
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module

from train import cli_main


@pytest.mark.parametrize(
    "model",
    [
        "electrolyte_fm.models.RoBERTa",
        "electrolyte_fm.models.RoBERTaPreLayerNorm",
    ],
)
def test_default(model):
    tag = f"testing-{randint(0, 128)}"
    with TemporaryDirectory() as fake_data_dir:
        cli = cli_main(
            [
                "--data",
                "RobertaDataSet",
                "--data.path",
                fake_data_dir,
                "--data.tokenizer",
                "ibm/MoLFormer-XL-both-10pct",
                "--model",
                "electrolyte_fm.models.RoBERTa",
                model,
                "--trainer.devices=1",
                f"--tags=['{tag}']",
            ]
        )
    assert cli.datamodule.vocab_size == cli.model.vocab_size
    assert cli.datamodule.vocab_size == len(cli.datamodule.tokenizer)
    assert cli.trainer.logger.__class__ == WandbLogger
    assert isinstance(cli.trainer.logger, WandbLogger)
    assert cli.trainer.logger._wandb_init["tags"] == [
        tag,
    ]


def test_finetune(monkeypatch):
    with TemporaryDirectory() as fake_data_dir:
        # Create a fake config
        ckpt = Path(fake_data_dir).joinpath(
            "fake-job", "checkpoints", "not-a-real.ckpt"
        )
        ckpt.mkdir(parents=True)
        with open(ckpt.parent.parent.joinpath("config.json"), "w") as fid:
            json.dump({"data": {"tokenizer": "smirk"}}, fid)
        with open(ckpt.parent.parent.joinpath("model_hparams.json"), "w") as fid:
            json.dump(
                {
                    "version": "0.2.0",
                    "class_path": "electrolyte_fm.models.roberta_base.RoBERTa",
                    "init_args": {"vocab_size": 128},
                },
                fid,
            )

        # Patch out loading so we don't need a real checkpoint
        monkeypatch.setattr(Module, "load_state_dict", lambda *args, **_: None)
        monkeypatch.setattr(
            zero_to_fp32,
            "get_fp32_state_dict_from_zero_checkpoint",
            lambda *args, **kwargs: None,
        )

        cli = cli_main(
            [
                "--data=PropertyPredictionDataModule",
                f"--data.path={fake_data_dir}",
                "--model=LMFinetuning",
                f"--model.encoder_ckpt={ckpt}",
                "--trainer.devices=1",
            ]
        )
        assert str(cli.datamodule.tokenizer.__class__)
