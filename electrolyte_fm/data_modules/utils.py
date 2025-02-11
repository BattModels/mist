from enum import Enum
from typing import Optional, TypeVar
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
    KEUKLE_SMILES = "smiles-keukle"


def encode_molecules(
    ds: AbstractDataset,
    input_column: str,
    output_column: Optional[str] = None,
    encoding: MolEncoding = MolEncoding.SMILES,
    **kwargs,
) -> AbstractDataset:
    """Convert SMILES encoding in `input_column` to desired `encoding` and save to `output_column`.
    Defaulting to overwriting the input column. Additional kwargs are passed to the encoding function
    and `ds.map`
    """
    assert isinstance(input_column, str)
    output_column = output_column or input_column

    if encoding == MolEncoding.SMILES:
        assert output_column == input_column
        return ds

    elif encoding == MolEncoding.SELFIES:
        return encode_selfies(ds, input_column, output_column, **kwargs)

    elif encoding == MolEncoding.CANONICAL_SMILES:
        return encode_canonical_smiles(ds, input_column, output_column, **kwargs)

    elif encoding == MolEncoding.KEUKLE_SMILES:
        return encode_keukle(ds, input_column, output_column, **kwargs)

    else:
        raise RuntimeError(f"Unknown encoding: {encoding}")


def encode_selfies(ds, input_column: str, output_column: str, **kwargs):
    """Encode SMILES to selfies, if possible, filtering out encoding failures"""
    ds = ds.map(
        lambda x: {output_column: _maybe_encode_selfies(x)},
        input_columns=input_column,
        batched=False,
        **kwargs,
    )
    ds = ds.filter(lambda x: x[output_column] is not None, batched=False, **kwargs)
    return ds


def _maybe_encode_selfies(smi: str) -> Optional[str]:
    import selfies

    try:
        return selfies.encoder(smi)
    except selfies.EncoderError:
        return None


def encode_canonical_smiles(
    ds, input_column: str, output_column: str, filter_failures: bool = False, **kwargs
):
    """Encode SMILES to canonical SMILES, if possible, filtering out encoding failures"""

    ds = ds.map(
        lambda x: {output_column: _maybe_encode_canonical_smiles(x)},
        input_columns=input_column,
        batched=False,
        **kwargs,
    )

    if filter_failures:
        ds = ds.filter(lambda x: x[output_column] is not None, batched=False, **kwargs)

    return ds


def _maybe_encode_canonical_smiles(
    smi: str, filter_failures: bool = False
) -> Optional[str]:
    try:
        return Chem.CanonSmiles(smi)
    except Exception:
        return None if filter_failures else smi


def encode_keukle(ds, input_column: str, output_column: str, **kwargs):
    """Encode SMILES to keukle form, if possible, filtering out encoding failures"""

    def _maybe_encode_keukle(smi: str) -> Optional[str]:
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)
        except Exception:
            return None

    ds = ds.map(
        lambda x: {output_column: _maybe_encode_keukle(x)},
        input_columns=input_column,
        batched=False,
        **kwargs,
    )
    ds = ds.filter(lambda x: x[output_column] is not None, batched=False, **kwargs)
    return ds
