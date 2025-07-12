from pathlib import Path

import torch
from datasets import Dataset, IterableDatasetDict, load_dataset
from torch.utils.data import default_collate

from .utils import MolEncoding, stack_columns
from .property_prediction_dataset import PropertyPredictionDataModule, collate_target


class ComponentDataModule(PropertyPredictionDataModule):
    def __init__(
        self,
        path: str | Path,
        n_components: int = 2,
        include_temperature: bool | str = True,
        smi_column: str = "smi{:d}",
        x_column: str = "x{:d}",
        excess_columns: list[str] | None = None,
        **kwargs,
    ):
        self.path = Path(path)
        self.include_temperature = bool(include_temperature)
        self.n_components = int(n_components)
        self.smi_columns = [smi_column.format(n + 1) for n in range(self.n_components)]
        self.x_columns = [x_column.format(n + 1) for n in range(self.n_components)]
        self.excess_columns = excess_columns or []
        assert self.path.exists()
        super().__init__(smi_column=smi_column, **kwargs)
        assert len(self.target_columns) >= 1
        assert len(self.smi_columns) == len(self.x_columns) == self.n_components

    def _get_dataset(self):
        ds = load_dataset(
            "arrow",
            name=self.path.name,
            data_files={
                "train": str(self.path.joinpath("train/*.arrow")),
                "validation": str(self.path.joinpath("validation/*.arrow")),
                "test": str(self.path.joinpath("test/*.arrow")),
            },
            keep_in_memory=False,
            streaming=True,
            save_infos=False,
        )
        assert isinstance(ds, IterableDatasetDict)
        return ds

    def setup(self, stage: str) -> None:
        # Prepare target columns
        ds = self.dataset
        ds = ds.map(
            collate_target,
            batched=False,
            fn_kwargs={"target_columns": self.target_columns},
            remove_columns=self.target_columns,
        )

        if self.excess_columns:
            ds = ds.map(
                collate_target,
                batched=False,
                fn_kwargs={
                    "target_columns": self.excess_columns,
                    "name": "target_excess",
                },
                remove_columns=self.excess_columns,
            )

        ds = ds.map(
            stack_columns,
            batched=True,
            fn_kwargs={"columns": self.x_columns, "output": "composition"},
            remove_columns=self.x_columns,
        )

        # Maybe encode and tokenize
        if not self.randomize:
            ds = ds.map(
                encode_and_tokenize_mixture,
                batched=True,
                fn_kwargs={
                    "smi_columns": self.smi_columns,
                    "randomize": self.randomize,
                    "encoding": self.encoding,
                    "tokenizer": self.tokenize,
                    "token_collator": self.token_collator,
                },
            )

        # Filter to input columns
        columns = ["target", "target_mask", "composition"]
        if self.include_encoding or self.randomize:
            columns.extend(self.smi_columns)
        if self.excess_columns:
            columns.extend(["target_excess", "target_excess_mask"])

        if not self.randomize:
            columns.extend(["input_ids", "attention_mask"])
        if self.include_temperature:
            columns.append("temperature")
        ds = ds.select_columns(columns)

        self.train_dataset: Dataset = ds["train"].shuffle()
        self.val_dataset: Dataset = ds["validation"]
        self.test_dataset: Dataset = ds["test"]
        self.target_dataset = ds["train"].select_columns(["target", "target_mask"])

    def collate_fn(self, batch):
        batch = default_collate(batch)
        batch["composition"] = default_collate(batch["composition"]).T
        if self.randomize:
            batch = encode_and_tokenize_mixture(
                batch,
                tokenizer=self.tokenize,
                encoding=self.encoding,
                smi_columns=self.smi_columns,
                randomize=self.randomize,
                token_collator=self.token_collator,
            )
            if not self.include_encoding:
                for smi in self.smi_columns:
                    batch.pop(smi)

        return batch


def encode_and_tokenize_mixture(
    batch: dict[str, list | torch.Tensor],
    tokenizer=None,
    encoding: MolEncoding = MolEncoding.SMILES,
    smi_columns: list[str] = [],
    randomize: bool = True,
    token_collator=None,
):
    encode = encoding.random if randomize else encoding
    B = len(batch[smi_columns[0]])
    smi = [encode(batch[col][bdx]) for bdx in range(B) for col in smi_columns]  # (B*N)

    toks = token_collator(tokenizer(smi))
    for k in ["input_ids", "attention_mask"]:
        batch[k] = torch.reshape(toks[k], (B, len(smi_columns), -1))

    return batch
