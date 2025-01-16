from pathlib import Path
from typing import List, Optional

from datasets import Dataset, load_dataset

from ..utils.tokenizer import load_tokenizer
from .molnet_dataset import MolNetDataModule
from .utils import MolEncoding


class PropertyPredictionDataModule(MolNetDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: str,
        batch_size: int = 64,
        num_workers: int = 1,
        prefetch_factor: int = 4,
        smi_column: str = "smiles",
        target_columns: List[str] = ["Class"],
        strip_unk_tokens: bool = False,
        val_batch_size: Optional[int] = None,
        encoding: str = "smiles",
    ):
        super().__init__()

        self.tokenizer = load_tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer)
        self.path: Path = Path(path)
        assert self.path.is_dir()

        self.smi_column = smi_column
        self.target_columns = target_columns
        self.strip_unk_tokens = strip_unk_tokens
        self.encoding = MolEncoding(encoding)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.hparams["vocab_size"] = self.vocab_size
        self.save_hyperparameters(logger=False)

    @property
    def dataset(self):
        if hasattr(self, "__dataset"):
            return self.__dataset

        # Load datasets, checking for splits
        ds = load_dataset(str(self.path), keep_in_memory=False, streaming=True)
        assert "train" in ds and "validation" in ds
        self.__dataset = ds
        return self.__dataset
