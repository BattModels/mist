import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from datasets import Dataset
from lightning.pytorch import LightningDataModule

from electrolyte_fm.data_modules import (
    RobertaDataSet,
    tmQMDataModule,
)
from electrolyte_fm.data_modules.molnet_dataset import strip_unk_tokens
from electrolyte_fm.data_modules.utils import MolEncoding, encode_molecules
from electrolyte_fm.utils.tokenizer import load_tokenizer


def check_datamodule(dm: LightningDataModule, stage="fit", limit_batches=100):
    dm.prepare_data()
    dm.setup(stage)
    check_dataloader(dm.train_dataloader(), limit_batches)
    check_dataloader(dm.val_dataloader(), limit_batches)
    check_dataloader(dm.test_dataloader(), limit_batches)


def check_dataloader(dl, limit_batches):
    for idx, batch in enumerate(dl):
        print(batch.keys())
        assert "input_ids" in batch
        assert "labels" in batch or "target" in batch
        assert isinstance(batch["target"], list)
        assert isinstance(batch["target_mask"], list)
        if idx >= limit_batches:
            break


def safe_copyfile(src, dst: Path):
    dst.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src, dst)


@pytest.fixture(scope="session")
def fake_dataset():
    smiles = [
        "CC[N+](C)(C)Cc1ccccc1Br",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CN(CCC1=CNC2=C1C=CC=C2)C",
        "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
    ]
    with TemporaryDirectory() as dir:
        smi = Path(dir, "smiles.txt")
        smi.write_text("\n".join(smiles * 10))
        for idx in range(9):
            safe_copyfile(smi, Path(dir, "data", "train", f"{idx}.txt"))
            safe_copyfile(smi, Path(dir, "data", "val", f"{idx}.txt"))
            safe_copyfile(smi, Path(dir, "data", "test", f"{idx}.txt"))
        yield dir


@pytest.mark.parametrize("encoding", ["smiles", "selfies", "smiles-canonical"])
def test_realspace_dataset(fake_dataset, encoding):
    dm = RobertaDataSet(fake_dataset, "smirk", encoding=encoding)
    check_datamodule(dm)


def test_canonical_dataset(fake_dataset):
    dm = RobertaDataSet(fake_dataset, "smirk", canonical=True)
    assert dm.encoding == MolEncoding.CANONICAL_SMILES


@pytest.mark.parametrize("encoding", [e.value for e in MolEncoding])
def test_encoding(encoding):
    ds = Dataset.from_dict(
        {"text": ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCCC"]}
    )
    encode_molecules(ds, "text", encoding=MolEncoding(encoding))

def has_tmQm():
    files = Path(__file__).parent.parent.joinpath("opt", "tmQM", "data")
    if files.exists():
         return files 
    return None


@pytest.mark.parametrize("encoding", ["smiles", "selfies", "smiles-canonical"])
@pytest.mark.skipif(has_tmQm() is None, reason="Skipping tmQM tests")
def test_tmQM_dataset(encoding):
    dm = tmQMDataModule(has_tmQm(), 
    target_columns= ['Electronic_E', 'Dispersion_E'], 
    tokenizer="smirk", 
    encoding=encoding)
    check_datamodule(dm)


def test_strip_unknown_tokens():
    tok = load_tokenizer("ibm/MoLFormer-XL-both-10pct-oov")
    encoding = tok("🚀C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]")
    assert tok.unk_token_id in encoding["input_ids"]
    assert len(encoding["input_ids"]) == 33
    strip_encoding = strip_unk_tokens(encoding, tok.unk_token_id)
    assert "input_ids" in strip_encoding
    assert "attention_mask" in strip_encoding
    assert tok.unk_token_id not in strip_encoding["input_ids"]
    assert len(strip_encoding["input_ids"]) == 32
    assert strip_encoding.pop("is_oov")
    for k, v in strip_encoding.items():
        assert len(v) == 32
