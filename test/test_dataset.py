import pytest
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from pytorch_lightning import Trainer, LightningDataModule
from electrolyte_fm.data_modules import RobertaDataSet


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
