from smirk import SmirkTokenizerFast
from smirk.labeler import TokenLabeler
from asyncio import Semaphore
from pathlib import Path
from datasets import IterableDataset, load_dataset, concatenate_datasets
from electrolyte_fm.data_modules.property_prediction_dataset import (
    PropertyPredictionDataModule,
)
from electrolyte_fm.data_modules.utils import MolEncoding, AbstractDataset


class DigitDataset(PropertyPredictionDataModule):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        assert self.path.exists()
        kwargs["target_columns"] = ["digit_type"]
        assert self.encoding != MolEncoding.SELFIES

    def _get_dataset(self) -> AbstractDataset:
        return load_dataset(
            "arrow",
            name=str(self.path.name),
            data_files={
                "train": str(self.path.joinpath("data/train/*.arrow")),
                "validation": str(self.path.joinpath("data/validation/*.arrow")),
                "test": str(self.path.joinpath("data/test/*.arrow")),
            },
            keep_in_memory=False,
            streaming=True,
            save_infos=False,
        )  # type: ignore


def build_probe_dataset():
    tmqm_path = Path(__file__).parent.parent.parent.parent.joinpath("tmQM", "data")
    realspace_path = Path("/nfs/turbo/coe-venkvis/mist/realspace_v4_dev2/")
    tmqm = load_dataset(
        "arrow",
        name=str(tmqm_path.name),
        data_files=str(tmqm_path.joinpath("data/validation/*.arrow")),
        keep_in_memory=False,
        streaming=True,
        save_infos=False,
        split="train",
    )
    assert isinstance(tmqm, IterableDataset)
    tmqm = tmqm.select_columns("smiles").rename_column("smiles", "smi")
    realspace = load_dataset(
        "text",
        name=str(realspace_path.name),
        data_files=str(realspace_path.joinpath("data/validation/*.txt")),
        keep_in_memory=False,
        streaming=True,
        save_infos=False,
        split="train",
    )
    assert isinstance(realspace, IterableDataset)
    realspace = realspace.select_columns("text").rename_column("tex", "smi")
    realspace = realspace.take(100_000)

    ds = concatenate_datasets([tmqm, realspace])
    tok = SmirkTokenizerFast()
    ds = ds.map(tok, input_columns="smiles")

    sem = Semaphore(64)

    tl = TokenLabeler(tok)

    async def async_token_label(input_ids: list[int]):
        async with sem:
            return {"token_type": list(tl.label_tokens(input_ids))}

    ds = ds.map(async_token_label, batched=False, input_columns="input_ids")
