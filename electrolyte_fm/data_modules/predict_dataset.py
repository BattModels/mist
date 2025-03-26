from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from electrolyte_fm.utils.tokenizer import load_tokenizer
from .utils import maybe_shard_dataset
from datasets import load_dataset


class PredictDataModule(LightningDataModule):
    def __init__(
        self,
        name_or_path: str,
        data_files=None,
        split: Optional[str] = None,
        tokenizer: str = "smirk",
        smi_column: str = "smi",
        batch_size: int = 256,
        num_workers=8,
        prefetch_factor: Optional[int] = None,
    ):
        super().__init__()
        self.name_or_path = name_or_path
        self.data_files = data_files
        self.split = split
        self.tokenizer = load_tokenizer(tokenizer)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor or (4 if num_workers > 0 else None)
        self.smi_column = smi_column

    def prepare_data(self) -> None:
        self.dataset

    @property
    def dataset(self):
        if not hasattr(self, "__dataset"):
            self.__dataset = load_dataset(
                self.name_or_path,
                data_files=self.data_files,
                split=self.split or "train",
                streaming=True,
                trust_remote_code=True,
            )
        return self.__dataset

    def setup(self, stage: str) -> None:
        ds = maybe_shard_dataset(self.trainer, self.dataset)
        ds = ds.map(self.tokenizer, input_columns=self.smi_column)
        self.predict_dataset = ds.select_columns(["input_ids", "attention_mask"])

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )
