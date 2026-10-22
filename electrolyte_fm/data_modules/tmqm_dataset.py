from pathlib import Path

from datasets import DatasetDict, IterableDatasetDict, load_from_disk
import torch

from .property_prediction_dataset import PropertyPredictionDataModule


class tmQMDataModule(PropertyPredictionDataModule):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        super().__init__(**kwargs)

    def _get_dataset(self):
        stored = load_from_disk(str(self.path), keep_in_memory=False)
        assert isinstance(stored, DatasetDict)
        ds = IterableDatasetDict(
            {split: dataset.to_iterable_dataset() for split, dataset in stored.items()}
        )
        assert isinstance(ds, IterableDatasetDict)
        return ds

    def collate_fn(self, batch, randomize: bool | None = None):
        for i in range(len(batch)):
            batch[i]["target"] = batch[i]["target"].tolist()
            batch[i]["target_mask"] = batch[i]["target_mask"].tolist()
        output = self.token_collator(batch)
        if self.target_columns:
            output["target"] = torch.stack([torch.tensor(x["target"]) for x in batch])
            output["target_mask"] = torch.stack(
                [torch.tensor(x["target_mask"]) for x in batch]
            )
            assert output["target"].shape == output["target_mask"].shape
        return output


class tmQMTrainValidationDataModule(tmQMDataModule):
    """tmQM training module that cannot construct or expose the test split."""

    def _get_dataset(self):
        stored = load_from_disk(str(self.path), keep_in_memory=False)
        assert isinstance(stored, DatasetDict)
        ds = IterableDatasetDict(
            {
                split: stored[split].to_iterable_dataset()
                for split in ("train", "validation")
            }
        )
        assert isinstance(ds, IterableDatasetDict)
        assert set(ds) == {"train", "validation"}
        return ds
