import pytest
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from pytorch_lightning import Trainer, LightningDataModule
from electrolyte_fm.data_modules import RobertaDataSet, PropertyPredictionDataModule
from electrolyte_fm.data_modules.property_prediction_dataset import strip_unk_tokens
from electrolyte_fm.utils.tokenizer import load_tokenizer


def check_datamodule(dm: LightningDataModule, stage="fit", limit_batches=100):
    dm.prepare_data()
    dm.setup(stage)
    check_dataloader(dm.train_dataloader(), limit_batches)
    check_dataloader(dm.val_dataloader(), limit_batches)
    check_dataloader(dm.test_dataloader(), limit_batches)


def check_dataloader(dl, limit_batches):
    for idx, batch in enumerate(dl):
        assert "input_ids" in batch
        assert "labels" in batch
        if idx >= limit_batches:
            break


def safe_copyfile(src, dst: Path):
    dst.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src, dst)


@pytest.fixture(scope="session")
def fake_dataset():
    smi = Path(__file__).parent.parent.joinpath("smirk", "test", "smiles.txt")
    with TemporaryDirectory() as dir:
        for idx in range(9):
            safe_copyfile(smi, Path(dir, "data", "train", f"{idx}.txt"))
            safe_copyfile(smi, Path(dir, "data", "val", f"{idx}.txt"))
            safe_copyfile(smi, Path(dir, "data", "test", f"{idx}.txt"))
        yield dir


def test_dataset(fake_dataset):
    dm = RobertaDataSet(fake_dataset, "smirk")
    check_datamodule(dm)


def test_canonical_dataset(fake_dataset):
    dm = RobertaDataSet(fake_dataset, "smirk", canonical=True)
    assert dm.canonical
    check_datamodule(dm)


def test_property_dataset():
    dm = PropertyPredictionDataModule(
        "/nfs/turbo/coe-venkvis/mist/molformer_ft_full/freesolv",
        "ibm/MoLFormer-XL-both-10pct-oov",
        target_columns=["calc"],
    )
    dm.prepare_data()
    dm.setup("fit")
    check_dataloader(dm.train_dataloader(), 5)


def test_strip_unknown_tokens():
    tok = load_tokenizer("ibm/MoLFormer-XL-both-10pct-oov")
    encoding = tok("🚀C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]")
    assert tok.unk_token_id in encoding["input_ids"]
    assert len(tok.unk_token_id) == 33
    strip_encoding = strip_unk_tokens(encoding)
    assert "input_ids" in strip_encoding
    assert "attention_mask" in strip_encoding
    assert tok.unk_token_id not in strip_encoding["input_ids"]
    assert len(strip_encoding["input_ids"]) == 32
    for k, v in strip_encoding.items():
        assert len(v) == 32
