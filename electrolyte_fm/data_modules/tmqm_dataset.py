from pathlib import Path

from datasets import IterableDatasetDict, load_dataset

from .property_prediction_dataset import PropertyPredictionDataModule


class tmQMDataModule(PropertyPredictionDataModule):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        super().__init__(**kwargs)

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
