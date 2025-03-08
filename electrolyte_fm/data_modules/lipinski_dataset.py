from pathlib import Path

import torch
from datasets import load_dataset
from rdkit.Chem import Lipinski, MolFromSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt

from .molnet_dataset import _URLS
from .property_prediction_dataset import PropertyPredictionDataModule
from .utils import AbstractDataset, MolEncoding, filter_invalid_smi
from .molnet_dataset import train_val_test_split


class LipinskiDataModule(PropertyPredictionDataModule):
    def __init__(self, name_or_path: str, **kwargs):
        self.name_or_path = name_or_path
        if not Path(name_or_path).exists():
            # Set default smi_column
            assert name_or_path in _URLS.keys()
            self.name_or_path = _URLS[name_or_path]
            kwargs["smi_column"] = (
                kwargs.get("smi_column", None) or "smiles"
                if name_or_path != "bace"
                else "mol"
            )
        assert isinstance(kwargs["smi_column"], str)
        kwargs["additonal_columns"] = [
            "probe_target",
            *kwargs.get("additonal_columns", []),
        ]
        super().__init__(**kwargs)
        assert self.encoding != MolEncoding.SELFIES

    def _get_dataset(self) -> AbstractDataset:
        # Load the dataset
        ds: AbstractDataset = load_dataset(
            "csv",
            name=self.name,
            data_files=[self.name_or_path],
            split="train",
            keep_in_memory=False,
            save_infos=False,
        )  # type: ignore

        ds = ds.select_columns(self.smi_column)
        ds = filter_invalid_smi(ds, self.smi_column)

        ds = ds.map(
            lipinki_rule_of_five,
            batched=False,
            fn_kwargs={"smi_column": self.smi_column},
        )
        return train_val_test_split(ds)

    def collate_fn(self, batch):
        output = super().collate_fn(batch)
        output = self.token_collator(batch)
        output["probe_target"] = torch.stack(
            [torch.tensor(x["probe_target"]) for x in batch]
        )

        return output


def lipinki_rule_of_five(x: dict, smi_column: str) -> dict:
    smi = x[smi_column]
    mol = MolFromSmiles(smi)
    assert mol is not None, "invalid smi: %s" % smi
    x["num_h_bond_donors"] = Lipinski.NumHDonors(mol)
    x["lipinki_h_donor"] = x["num_h_bond_donors"] <= 5
    x["num_h_bond_acceptors"] = Lipinski.NumHAcceptors(mol)
    x["lipinki_h_acceptor"] = x["num_h_bond_acceptors"] <= 10
    x["molecular_weight"] = ExactMolWt(mol)
    x["lipinki_mwt"] = x["molecular_weight"] <= 500
    x["log_p"] = MolLogP(mol)
    x["lipinki_log_p"] = x["log_p"] <= 5
    x["lipinki"] = all(v for k, v in x.items() if k.startswith("lipinki"))
    x["probe_target"] = [v for k, v in x.items() if k.startswith("lipinki")]
    return x


if __name__ == "__main__":
    ds = LipinskiDataModule(name_or_path="hiv")
    ds = ds.dataset
    df = ds.to_pandas()
    df.to_csv("lipo.csv")
    cols = [
        "lipinki",
        "lipinki_h_donor",
        "lipinki_h_acceptor",
        "lipinki_mwt",
        "lipinki_log_p",
    ]
    print({k: df[k].value_counts() for k in cols})
