from abc import abstractmethod
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from datasets import Dataset
from transformers import DataCollatorWithPadding
from functools import partial
from threading import Lock

from ..utils.tokenizer import load_tokenizer
from .utils import (
    MolEncoding,
    AbstractDataset,
    maybe_shard_dataset,
    encode_molecules,
    collate_target,
    is_fast,
    deterministic_evaluation_encoding,
)

_TOKENIZER_LOCK = Lock()


def tokenize_serially(tokenize, *args, **kwargs):
    """Protect mutable Rust-backed tokenizer configuration across shard readers."""
    with _TOKENIZER_LOCK:
        return tokenize(*args, **kwargs)


class PropertyPredictionDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: str = "smirk",
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        smi_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        encoding: str = MolEncoding.SMILES.value,
        include_encoding: bool = False,
        randomize: bool = False,
        stateless_randomize: bool = False,
        truncation: bool = False,
        max_seq_length: int = 510,
    ):
        super().__init__()

        self.tokenizer = load_tokenizer(tokenizer)
        self.token_collator = DataCollatorWithPadding(self.tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.truncation = truncation
        self.max_length = max_seq_length
        tokenize = partial(
            self.tokenizer, truncation=self.truncation, max_length=self.max_length
        )
        # Streaming datasets read Arrow shards concurrently. Tokenizers such as
        # Smirk mutate their truncation configuration per call.
        self.tokenize = partial(tokenize_serially, tokenize)

        self.smi_column = smi_column
        self.target_columns = target_columns
        self.encoding = MolEncoding(encoding)
        self.include_encoding = include_encoding
        self.randomize = randomize
        self.stateless_randomize = stateless_randomize
        if stateless_randomize:
            from electrolyte_fm.utils.callbacks.stateless_augmentation import shared_epoch
            self.augmentation_epoch=shared_epoch()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor or (4 if num_workers > 0 else None)
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        self.__dataset = self._get_dataset()
        return self.__dataset

    @abstractmethod
    def _get_dataset(self) -> AbstractDataset:
        raise NotImplementedError()

    def setup(self, stage: str) -> None:
        # Load datasets, checking for splits
        ds = self.dataset
        ds = maybe_shard_dataset(self.trainer, ds)
        # Preserve the declared training representation, but never let raw
        # source atom order leak into model selection or sealed evaluation.
        # ``smiles`` historically means identity in MolEncoding, so evaluation
        # must opt into the canonical enum explicitly.
        ds = ds.__class__({
            split: encode_molecules(
                value,
                self.smi_column,
                encoding=(deterministic_evaluation_encoding(self.encoding) if split != "train" else self.encoding),
            )
            for split, value in ds.items()
        })

        # Remove extraneous columns and tokenize smiles
        if targets := self.target_columns:
            ds = ds.map(
                collate_target,
                batched=False,
                fn_kwargs={"target_columns": targets},
                remove_columns=targets,
            )

            # Save training dataset for target transformations
            self.target_dataset = ds["train"].select_columns(["target", "target_mask"])
            ds = ds.select_columns([self.smi_column, "target", "target_mask"])
        else:
            ds = ds.select_columns([self.smi_column])

        # Tokenize
        ds = ds.map(
            self.tokenize,
            batched=is_fast(self.tokenizer),
            input_columns=self.smi_column,
        )
        if not self.include_encoding and not self.randomize:
            ds = ds.remove_columns(self.smi_column)

        self.train_dataset: Dataset = ds["train"].shuffle(seed=42)
        self.val_dataset: Dataset = ds["validation"]
        # Validation-only sweeps intentionally omit the sealed test split.
        # ``fit`` must not require or materialize it; ``test`` will fail
        # explicitly through the absent loader dataset if invoked.
        self.test_dataset: Optional[Dataset] = ds.get("test")
        self.token_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")

    def collate_fn(self, batch, randomize: bool | None = None):
        tokenizer = self.tokenize
        encoding = self.encoding
        if self.randomize if randomize is None else randomize:
            for idx in range(len(batch)):
                if getattr(self,'stateless_randomize',False):
                    from electrolyte_fm.utils.callbacks.stateless_augmentation import deterministic_random_smiles
                    new_smi=deterministic_random_smiles(encoding,batch[idx][self.smi_column],self.__class__.__name__,self.augmentation_epoch.value)
                else:
                    new_smi = encoding.random(batch[idx][self.smi_column])
                if not getattr(self,'stateless_randomize',False):
                    try:
                        from replication.c1_randomization import record_randomization
                        record_randomization(getattr(encoding,'value',str(encoding)), batch[idx][self.smi_column])
                    except ImportError:
                        pass
                if new_smi is not None:
                    batch[idx].update(tokenizer(new_smi))

        if not self.include_encoding:
            # The source representation is needed to regenerate randomized
            # training encodings, but it is never a tokenizer tensor field.
            # Remove it for canonical validation/test batches as well.
            for row in batch:
                row.pop(self.smi_column, None)

        targets = None
        target_masks = None
        if self.target_columns:
            # Targets are model inputs, not tokenizer features.  Remove them
            # before Hugging Face padding so tensor-valued target vectors are
            # never interpreted as variable-length token fields.
            targets = torch.stack([torch.as_tensor(x.pop("target")) for x in batch])
            target_masks = torch.stack(
                [torch.as_tensor(x.pop("target_mask")) for x in batch]
            )

        output = self.token_collator(batch)
        if targets is not None and target_masks is not None:
            output["target"] = targets
            output["target_mask"] = target_masks
            assert output["target"].shape == output["target_mask"].shape

        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=partial(self.collate_fn, randomize=self.randomize),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=partial(self.collate_fn, randomize=False),
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test split is sealed and unavailable in this data module")
        return DataLoader(
            self.test_dataset,
            collate_fn=partial(self.collate_fn, randomize=False),
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )
