import logging
import random
from asyncio import Semaphore
from enum import Enum
from typing import Optional, TypeVar

from datasets import Dataset, DatasetDict, IterableDatasetDict
from datasets.distributed import split_dataset_by_node
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupShuffleSplit


def is_fast(tokenizer):
    """Helper for check if something is a fast tokenizer"""
    if hasattr(tokenizer, "is_fast"):
        return tokenizer.is_fast
    return False


AbstractDataset = TypeVar("AbstractDataset", Dataset, DatasetDict, IterableDatasetDict)


def maybe_shard_dataset(trainer, ds: AbstractDataset) -> AbstractDataset:
    """Maybe shard a dataset across trainer ranks, if appropriate"""
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        return ds.__class__({k: maybe_shard_dataset(trainer, v) for k, v in ds.items()})
    if trainer is None:
        return ds
    return split_dataset_by_node(ds, trainer.global_rank, trainer.world_size)


class MolEncoding(Enum):
    """Enumeration of supported molecule encodings"""

    SMILES = "smiles"
    SELFIES = "selfies"
    CANONICAL_SMILES = "smiles-canonical"
    KEKULE = "smiles-kekule"

    def __call__(self, smi: str):
        if self == MolEncoding.SMILES:
            return smi

        elif self == MolEncoding.SELFIES:
            import selfies

            try:
                return selfies.encoder(smi)
            except selfies.EncoderError:
                return None

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        if self == MolEncoding.CANONICAL_SMILES:
            return Chem.MolToSmiles(mol, canonical=True)
        elif self == MolEncoding.KEKULE:
            return Chem.MolToSmiles(mol, kekuleSmiles=True)

        assert False, "Not Reachable, missing Enum Branch"

    def random(self, smi: str):
        if self == MolEncoding.SELFIES:
            return self(smi)  # Randomization not possible

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        if self == MolEncoding.SMILES:
            return Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                kekuleSmiles=(random.random() > 0.5),
            )
        elif self == MolEncoding.KEKULE:
            return Chem.MolToSmiles(
                mol, canonical=False, kekuleSmiles=True, doRandom=True
            )
        elif self == MolEncoding.CANONICAL_SMILES:
            return Chem.MolToSmiles(mol, canonical=True, doRandom=True)

        assert False, "Not Reachable, missing Enum Branch"


def encode_molecules(
    ds: AbstractDataset,
    input_column: str,
    output_column: Optional[str] = None,
    encoding: MolEncoding = MolEncoding.SMILES,
    random: bool = False,
    max_workers: int = 8,
    **kwargs,
) -> AbstractDataset:
    """Convert SMILES encoding in `input_column` to desired `encoding` and save to `output_column`.
    Defaulting to overwriting the input column. Additional kwargs are passed to the encoding function
    and `ds.map`
    """
    assert isinstance(input_column, str)
    output_column = output_column or input_column
    encode = encoding if not random else encoding.random

    tasks = Semaphore(max_workers)

    async def async_encode(batch: list[str]) -> dict:
        async with tasks:
            return {output_column: [encode(smi) for smi in batch]}

    async def async_filter(batch: list[str | None]) -> list[bool]:
        async with tasks:
            return [x is not None for x in batch]

    ds = ds.map(
        async_encode,
        input_columns=input_column,
        batched=True,
        **kwargs,
    )
    return ds.filter(async_filter, batched=True, input_columns=output_column, **kwargs)


def train_val_test_split(ds, **kwargs):
    ds_train_other = ds.train_test_split(test_size=0.2, seed=42, **kwargs)
    ds_val_test = ds_train_other["test"].train_test_split(
        test_size=0.5, seed=42, **kwargs
    )
    return DatasetDict(
        {
            "train": ds_train_other["train"],
            "validation": ds_val_test["train"],
            "test": ds_val_test["test"],
        }
    )


def scaffold_hash(smi: str) -> str:
    try:
        scaffold = MurckoScaffoldSmiles(smi)
    except ValueError:
        logging.warning("No scaffold for %s, using input smiles string", smi)
        scaffold = smi
    return scaffold


def scaffold_split(ds: Dataset, smi_column):
    # Hash scaffolds and then bin into groups, maintains the scaffold split
    # but reduces the compute
    df = ds.map(
        lambda x: {"scaffold": scaffold_hash(x)},
        input_columns=smi_column,
        batched=False,
    ).to_pandas(batched=False)

    # Split
    train, other = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            df.index, groups=df["scaffold"].values
        )
    )
    df_other = df.iloc[other]
    val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(
            df_other, groups=df.iloc[other]["scaffold"]
        )
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(df.iloc[train], preserve_index=False),
            "validation": Dataset.from_pandas(df_other.iloc[val], preserve_index=False),
            "test": Dataset.from_pandas(df_other.iloc[test], preserve_index=False),
        }
    )


def strip_unk_tokens(encoding: dict, unk_token_id: int) -> dict:
    """Remove unknown tokens from input"""
    is_oov = [id == unk_token_id for id in encoding["input_ids"]]
    out = {}
    for k, v in encoding.items():
        assert len(v) == len(is_oov)
        out[k] = [x for x, oov in zip(v, is_oov) if not oov]
    out["is_oov"] = any(is_oov)
    return out
