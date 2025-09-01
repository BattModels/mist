from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from .tokenizer import load_tokenizer

from .utils import (
    MolEncoding,
    AbstractDataset,
    maybe_shard_dataset,
    encode_molecules,
    is_fast,
    scaffold_split,
    train_val_test_split,
)


@dataclass
class GlobalComm:
    global_rank: int = 0
    world_size: int = 1


MOLNET_URLS = {
    "qm8": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
    "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
    "lipo": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "muv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "toxcast": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
    "sider": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
}


def tokenizer_dataset(
    tokenizer: str,
    name_or_path: str,
    encoding: str,
    world_size: int = 1,
    global_rank: int = 0,
    limit: int | None = None,
    max_workers: int = 8,
) -> AbstractDataset:
    tokenizer = load_tokenizer(tokenizer)
    encoding = MolEncoding(encoding)
    ds = get_dataset(name_or_path)
    ds = maybe_shard_dataset(GlobalComm(global_rank, world_size), ds)

    ds = encode_molecules(ds, "smi", encoding=encoding, max_workers=max_workers)
    ds = ds.map(
        tokenizer,
        batched=is_fast(tokenizer),
        input_columns="smi",
    )

    if limit is not None:
        return ds.shuffle(seed=42).take(limit)
    else:
        return ds


def get_dataset(name_or_path: str):
    if Path(name_or_path).is_dir():
        if "tmQM" in Path(name_or_path).parts:
            return tmqm(name_or_path)
        else:
            return smiles_dataset(name_or_path)
    else:
        assert name_or_path in MOLNET_URLS.keys()
        return molnet(name_or_path)


def molnet(name: str):
    ds = load_dataset(
        "csv",
        name=name,
        data_files=[MOLNET_URLS[name]],
        split="train",
        streaming=False,
        save_infos=False,
    )
    ds = ds.rename_column("smiles" if name != "bace" else "mol", "smi")
    ds = ds.select_columns("smi")

    if name in ["hiv", "bace", "bbbp"]:
        return scaffold_split(ds, "smi")
    else:
        return train_val_test_split(ds)


def tmqm(name_or_path: str):
    path = Path(name_or_path)
    ds = load_dataset(
        "arrow",
        name=path.name,
        data_files={
            "train": str(path.joinpath("train/*.arrow")),
            "validation": str(path.joinpath("validation/*.arrow")),
            "test": str(path.joinpath("test/*.arrow")),
        },
        keep_in_memory=False,
        streaming=True,
        save_infos=False,
    )
    return ds.select_columns("smiles").rename_column("smiles", "smi")


def smiles_dataset(name_or_path: str):
    path = Path(name_or_path)
    ds = load_dataset(
        "text",
        name=str(path.name),
        data_files={
            "train": str(path.joinpath("data/train/*.txt")),
            "validation": str(path.joinpath("data/val/*.txt")),
            "test": str(path.joinpath("data/test/*.txt")),
        },
        keep_in_memory=False,
        streaming=True,
        save_infos=False,
    )
    return ds.rename_column("text", "smi")
