from datasets import Dataset, load_dataset

from .property_prediction_dataset import PropertyPredictionDataModule
from .utils import scaffold_split, train_val_test_split

_URLS = {
    "qm8": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
    "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
    "lipo": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "muv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "toxcast": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
    "sider": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
}


class MolNetDataModule(PropertyPredictionDataModule):
    def __init__(
        self,
        name: str = "bace",
        split: str = "random",
        **kwargs,
    ):
        # Set default smi_column
        kwargs["smi_column"] = (
            kwargs.get("smi_column", None) or "smiles" if name != "bace" else "mol"
        )
        assert isinstance(kwargs["smi_column"], str)

        self.name = name
        self.split = split
        super().__init__(**kwargs)

    def prepare_data(self):
        # Fetch data from the head node
        self.dataset

    def _get_dataset(self):
        # Load the dataset
        ds: Dataset = load_dataset(
            "csv",
            name=self.name,
            data_files=[_URLS[self.name]],
            split="train",
            keep_in_memory=False,
            save_infos=False,
        )  # type: ignore

        if self.name == "qm8":
            # Rename qm8 columns to remove duplicates and include source theory
            ds = ds.rename_columns(
                {
                    "E1-CC2": "E1-CC2-RI-CC2/def2TZVP",
                    "E2-CC2": "E2-CC2-RI-CC2/def2TZVP",
                    "f1-CC2": "f1-CC2-RI-CC2/def2TZVP",
                    "f2-CC2": "f2-CC2-RI-CC2/def2TZVP",
                    "E1-PBE0": "E1-PBE0-LR-TDPBE0/def2SVP",
                    "E2-PBE0": "E2-PBE0-LR-TDPBE0/def2SVP",
                    "f1-PBE0": "f1-PBE0-LR-TDPBE0/def2SVP",
                    "f2-PBE0": "f2-PBE0-LR-TDPBE0/def2SVP",
                    "E1-PBE0.1": "E1-PBE0-LR-TDPBE0/def2TZVP",
                    "E2-PBE0.1": "E2-PBE0-LR-TDPBE0/def2TZVP",
                    "f1-PBE0.1": "f1-PBE0-LR-TDPBE0/def2TZVP",
                    "f2-PBE0.1": "f2-PBE0-LR-TDPBE0/def2TZVP",
                    "E1-CAM": "E1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "E2-CAM": "E2-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "f1-CAM": "f1-CAM-LR-TDCAM-B3LYP/def2TZVP",
                    "f2-CAM": "f2-CAM-LR-TDCAM-B3LYP/def2TZVP",
                }
            )

        # Spit into train/val/test
        if self.split == "scaffold":
            return scaffold_split(ds, self.smi_column)
        elif self.split == "random":
            return train_val_test_split(ds)
        else:
            raise ValueError(f"Unknown split {self.split}")
