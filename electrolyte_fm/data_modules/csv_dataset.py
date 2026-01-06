from .property_prediction_dataset import PropertyPredictionDataModule
from .molnet_dataset import train_val_test_split
from datasets import Dataset, load_dataset
from pathlib import Path


class CSVDataModule(PropertyPredictionDataModule):
    def __init__(self, path: str, **kwargs):
        # Set default smi_column
        self.path = path
        assert Path(self.path).is_file()
        super().__init__(**kwargs)

    def prepare_data(self):
        # Fetch data from the head node
        self.dataset

    def _get_dataset(self):
        # Load the dataset
        ds: Dataset = load_dataset(
            "csv",
            data_files=[self.path],
            split="train",
            keep_in_memory=False,
            save_infos=False,
        )  # type: ignore

        return train_val_test_split(ds)
