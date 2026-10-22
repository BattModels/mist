"""Load the LR sweep's train/validation CSVs without opening the sealed test set."""

from pathlib import Path

from datasets import DatasetDict, load_dataset

from electrolyte_fm.data_modules.property_prediction_dataset import (
    PropertyPredictionDataModule,
)


class SplitCSVDataModule(PropertyPredictionDataModule):
    def __init__(self, split_dir: str, test_csv: str | None = None, **kwargs):
        self.split_dir = split_dir
        self.test_csv = test_csv
        if not Path(split_dir).is_dir():
            raise FileNotFoundError(split_dir)
        super().__init__(**kwargs)

    def _get_dataset(self) -> DatasetDict:
        files = {
            x: str(Path(self.split_dir, f"{x}.csv")) for x in ("train", "validation")
        }
        # Evaluation opts in only after the validation-only choice is fixed.
        if self.test_csv is not None:
            files["test"] = self.test_csv
        if not all(Path(x).is_file() for x in files.values()):
            raise FileNotFoundError(self.split_dir)
        # Parallel LR cells otherwise write the same derived Arrow cache files.
        # These prepared benchmark tables fit in memory; keep their maps local
        # to each process while preserving the supplied split membership.
        return load_dataset("csv", data_files=files, keep_in_memory=True)
