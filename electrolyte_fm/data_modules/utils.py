import asyncio
from enum import Enum
import random
from typing import TypeVar
import torch
from rdkit import Chem
from datasets import Dataset, DatasetDict, IterableDatasetDict
from datasets.distributed import split_dataset_by_node


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


def filter_invalid_smi(
    ds: AbstractDataset, input_column: str, max_procs: int = 20, **kwargs
) -> AbstractDataset:
    sem = asyncio.Semaphore(max_procs)

    async def is_valid(x: dict):
        async with sem:
            mol = Chem.MolFromSmiles(x[input_column])
            return mol is not None

    return ds.filter(is_valid, batched=False, **kwargs)


def encode_molecules(
    ds: AbstractDataset,
    input_column: str,
    output_column: str | None = None,
    encoding: MolEncoding = MolEncoding.SMILES,
    random: bool = False,
    **kwargs,
) -> AbstractDataset:
    """Convert SMILES encoding in `input_column` to desired `encoding` and save to `output_column`.
    Defaulting to overwriting the input column. Additional kwargs are passed to the encoding function
    and `ds.map`
    """
    assert isinstance(input_column, str)
    output_column = output_column or input_column

    encode = encoding if not random else encoding.random
    ds = ds.map(
        lambda smi: {output_column: encode(smi)},
        input_columns=input_column,
        batched=False,
        **kwargs,
    )
    return ds.filter(lambda x: x[output_column] is not None, batched=False, **kwargs)


def stack_columns(batch, columns: list[str], output: str, dtype=None):
    n = len(batch[columns[0]])
    if dtype is None:
        convert = torch.tensor
    else:
        convert = lambda x: torch.tensor(x, dtype=dtype)
    return {output: [convert([batch[col][i] for col in columns]) for i in range(n)]}


def collate_target(x, target_columns, name: str = "target", dtype=None):
    """Stack multiple target columns into a single vector,
    recording unknown elements to be masked out during training
    """
    target = []
    mask = []
    for k in target_columns:
        v = x[k]
        if v is None:
            target.append(
                torch.tensor(0, dtype=dtype)
            )  # Placeholder, should be masked out
            mask.append(torch.tensor(False))
        else:
            target.append(torch.tensor(v, dtype=dtype))
            mask.append(torch.tensor(True))

    return {name: torch.stack(target), f"{name}_mask": torch.stack(mask)}
