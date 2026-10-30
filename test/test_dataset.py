import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import get_ident

import pytest
import torch
from datasets import Dataset, IterableDataset, IterableDatasetDict
from lightning.pytorch import LightningDataModule
from smirk import SmirkTokenizerFast

from electrolyte_fm.data_modules import (
    RobertaDataSet,
    tmQMDataModule,
)
from electrolyte_fm.data_modules.utils import (
    MolEncoding,
    encode_molecules,
    strip_unk_tokens,
)
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
        assert "labels" in batch or "target" in batch
        for k, v in batch.items():
            assert isinstance(k, str)
            assert isinstance(v, torch.Tensor)
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


@pytest.fixture(params=[e.value for e in MolEncoding])
def mol_encoding(request):
    return MolEncoding(request.param)


def test_encode(mol_encoding):
    assert isinstance(mol_encoding("O(c1cc(cc(OC)c1OC)CCN)C"), str)
    # Invalid Smiles are rejected if transcoded
    invalid = mol_encoding("Not A 🙂 String")
    if mol_encoding != MolEncoding.SMILES:
        assert invalid is None
    else:
        assert isinstance(invalid, str)


def test_random(mol_encoding):
    assert isinstance(mol_encoding.random("O=C1c2ccccc2C(=O)N1C3CCC(=O)NC3=O"), str)
    # Invalid Smiles are rejected if transcoded
    assert mol_encoding.random("Not A 🙂 String") is None


def test_realspace_dataset(fake_dataset, mol_encoding):
    dm = RobertaDataSet(fake_dataset, "smirk", encoding=mol_encoding)
    check_datamodule(dm)


def test_canonical_dataset(fake_dataset):
    dm = RobertaDataSet(fake_dataset, "smirk", canonical=True)
    assert dm.encoding == MolEncoding.CANONICAL_SMILES


@pytest.mark.parametrize("random", [True, False])
def test_encoding(mol_encoding, random):
    ds = Dataset.from_dict(
        {"text": ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCCC"]}
    )
    encode_molecules(ds, "text", encoding=mol_encoding, random=random)


def has_tmQm():
    files = Path(__file__).parent.parent.joinpath("opt", "tmQM", "data")
    if files.exists():
        return files
    return None


@pytest.mark.skipif(has_tmQm() is None, reason="Skipping tmQM tests")
def test_tmQM_dataset(mol_encoding):
    dm = tmQMDataModule(
        has_tmQm(),
        target_columns=["Electronic_E", "Dispersion_E"],
        tokenizer="smirk",
        encoding=mol_encoding,
    )
    check_datamodule(dm)


def test_streaming_shuffle_keeps_tokenization_out_of_prefetch_threads(monkeypatch):
    tokenization_threads = set()

    class RecordingTokenizer(SmirkTokenizerFast):
        def __call__(self, *args, **kwargs):
            tokenization_threads.add(get_ident())
            return super().__call__(*args, **kwargs)

    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(
        "electrolyte_fm.data_modules.property_prediction_dataset.load_tokenizer",
        lambda name: tokenizer,
    )

    def generate(shards):
        for shard in shards:
            for offset in range(4):
                value = shard * 4 + offset
                yield {"smiles": "C" * (value + 1), "energy": float(value)}

    class StreamingTMQM(tmQMDataModule):
        def _get_dataset(self):
            return IterableDatasetDict(
                {
                    split: IterableDataset.from_generator(
                        generate, gen_kwargs={"shards": list(range(4))}
                    )
                    for split in ("train", "validation", "test")
                }
            )

    dm = StreamingTMQM(path="unused", tokenizer="smirk", target_columns=["energy"])
    dm.setup("fit")
    rows = list(dm.train_dataset)

    assert len(rows) == 16
    assert sorted(float(row["target"][0]) for row in rows) == list(range(16))
    assert all("input_ids" in row and "attention_mask" in row for row in rows)
    assert tokenization_threads == {get_ident()}


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


def test_randomize(mol_encoding):
    mols = set()
    for _ in range(10):
        mols.add(mol_encoding.random("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))

    if mol_encoding == MolEncoding.SELFIES:
        assert len(mols) == 1
        pytest.xfail("random selfies are not supported")
    else:
        assert len(mols) > 1
