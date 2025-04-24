import asyncio
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
from rdkit.Chem import Lipinski, MolFromSmiles, MolToInchiKey
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt

from .molnet_dataset import MolNetDataModule
from .property_prediction_dataset import PropertyPredictionDataModule
from .utils import AbstractDataset, MolEncoding, filter_invalid_smi


class LipinskiDataModule(PropertyPredictionDataModule):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        assert self.path.exists()

        kwargs["target_columns"] = [
            "lipinski_h_donor",
            "lipinski_h_acceptor",
            "lipinski_mwt",
            "lipinski_log_p",
            "lipinski",
        ]
        kwargs["smi_column"] = kwargs.get("smi_column", "smi")
        super().__init__(**kwargs)
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


def get_molnet_dataset(name: str):
    ds = MolNetDataModule(name=name, split="all")
    return ds._get_dataset().rename_column(ds.smi_column, "smi").select_columns("smi")


def build_probe_dataset():
    sem = asyncio.Semaphore(64)

    ds = [
        get_molnet_dataset(name)
        for name in [
            "hiv",
            "toxcast",
            "tox21",
            "clintox",
            "bbbp",
            "qm9",
            "qm8",
            "freesolv",
            "lipo",
            "muv",
            "sider",
            "esol",
        ]
    ]

    # # Pull molecules from Zinc
    # zinc_path = "/lustre/fs0/shared/zinc_v1"
    # ds_zinc = load_dataset(
    #     "text",
    #     name="zinc_train",
    #     data_files=str(Path(zinc_path).joinpath("data/train/*.txt")),
    #     split="train",
    #     keep_in_memory=False,
    #     save_infos=False,
    # )
    # ds_zinc = ds_zinc.take(1_000_000)
    #
    # def zinc_smi(text: str):
    #     return {"smi": text.split(" ")[0]}
    #
    # ds_zinc = ds_zinc.map(zinc_smi, batched=False, input_columns="text")
    # ds.append(ds_zinc)

    ds = concatenate_datasets(ds)
    ds = filter_invalid_smi(ds, "smi")

    async def async_lipinski_rule_of_five(x):
        async with sem:
            return lipinski_rule_of_five(x, "smi")

    ds = ds.map(async_lipinski_rule_of_five, batched=False)

    # Label with inchi key for de-duplication
    async def inchi_key(smi: str):
        async with sem:
            return {"inchi_key": MolToInchiKey(MolFromSmiles(smi))}

    ds = ds.map(inchi_key, batched=False, input_columns="smi")

    # Resample to balanced classes
    df = ds.to_pandas()
    df.drop_duplicates(subset="inchi_key", inplace=True)
    lip_cols = [
        "lipinski_h_donor",
        "lipinski_h_acceptor",
        "lipinski_mwt",
        "lipinski_log_p",
    ]

    # Rebalance and report stats
    print("Group Counts:\n", get_group_sizes(df, lip_cols))
    print("Dataset size:", len(df))
    print("Class Odds:\n", df[lip_cols].mean())
    df = downsample_ipf_binary(df, lip_cols, n_samples=10_000)
    print("Group Counts:\n", get_group_sizes(df, lip_cols))
    print("Dataset size:", len(df))
    print("Class Odds:\n", df[lip_cols].mean())

    # Split preserving the frequency of each subgroup
    spliter = StratifiedShuffleSplit(train_size=0.80, random_state=721153)
    train_idx, test_idx = next(spliter.split(df["smi"], df["lipinski"]))

    # Save to disk
    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(df.iloc[train_idx], preserve_index=False),
            "validation": Dataset.from_pandas(df.iloc[test_idx], preserve_index=False),
        }
    )
    Path("lipinski").mkdir(exist_ok=True)
    ds.save_to_disk("lipinski/data", num_shards={"train": 8, "validation": 8})


def get_group_sizes(df: pd.DataFrame, class_columns) -> pd.DataFrame:
    """
    Get the count of each unique combination of values in the given class_columns.

    Parameters:
    - df: pandas DataFrame.
    - class_columns: A string or list of columns to group by.

    Returns:
    - A DataFrame showing each group and its count, sorted descending.
    """
    if isinstance(class_columns, str):
        class_columns = [class_columns]

    group_counts = df.groupby(class_columns).size().reset_index(name="count")
    return group_counts.sort_values("count", ascending=False).reset_index(drop=True)


def downsample_ipf_binary(
    df: pd.DataFrame,
    class_columns: list[str],
    n_samples: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Downsample a DataFrame with K binary columns so that each column’s marginal is 50/50,
    using iterative proportional fitting over the joint 2^K table.

    Steps:
      1. Build original cell counts for every combination of the K binaries.
      2. Initialize target cell counts = original counts.
      3. For each column, rescale all cell counts in each level (0 and 1) so that
         sum_over_cells(level=1) == total/2 and sum_over_cells(level=0) == total/2.
      4. Iterate until all K marginals are within tol of 0.5.
      5. Scale target cell counts to sum to n_samples (or len(df) if n_samples is None).
      6. For each cell c, weight per row in c = target_count[c] / original_count[c].
      7. Draw without replacement using these per‑row probabilities.

    Returns:
      A new DataFrame of size n_samples with approximately perfect 50/50 marginals.
    """
    rng = np.random.default_rng(random_state)
    df = df.reset_index(drop=True)
    N = len(df)

    if n_samples is None:
        min_count_factor = 4
        min_true_counts = [df[col].sum() for col in class_columns]
        min_false_counts = [len(df) - x for x in min_true_counts]
        max_true_size = int(min(min_true_counts) * min_count_factor)
        max_false_size = int(min(min_false_counts) * min_count_factor)
        n_samples = min(len(df), max_true_size, max_false_size)

    # 1) compute original contingency table
    #    use tuple of column values as key
    keys = list(df[class_columns].itertuples(index=False, name=None))
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    orig_counts = pd.Series(np.bincount(inv), index=range(len(uniq)), dtype=float)

    # initialize target = original
    target = orig_counts.copy()

    # precompute for each column which cells have bit=1
    # uniq is array of shape (n_cells, K)
    uniq_arr = np.array(uniq, dtype=int)
    is_one = {col: uniq_arr[:, i] == 1 for i, col in enumerate(class_columns)}

    total = target.sum()
    half = total / 2.0

    # 2) IPF loop
    for _ in range(max_iter):
        max_diff = 0.0

        for i, col in enumerate(class_columns):
            mask1 = is_one[col]
            mask0 = ~mask1

            # current marginal for this column
            cur1 = target[mask1].sum()
            cur0 = target[mask0].sum()

            # scale factor to push cur1 -> half and cur0 -> half
            if cur1 > 0:
                target[mask1] *= half / cur1
            if cur0 > 0:
                target[mask0] *= half / cur0

            # track worst marginal error
            max_diff = max(max_diff, abs(cur1 / total - 0.5), abs(cur0 / total - 0.5))

        if max_diff < tol:
            break
    else:
        # warn if not converged
        print(
            f"IPF did not converge in {max_iter} iters; max marginal error {max_diff:.2e}"
        )

    # 3) scale target total to n_samples
    scale = n_samples / total
    target *= scale

    # 4) per‐row weights: target_count[c] / orig_count[c]
    #    inv maps each row to its cell index
    cell_weight = target.to_numpy() / orig_counts.to_numpy()
    row_weights = cell_weight[inv]
    row_weights = np.clip(row_weights, 0, None)
    row_weights = row_weights / row_weights.sum()

    # 5) sample
    chosen = rng.choice(N, size=n_samples, replace=False, p=row_weights)
    return df.iloc[chosen].reset_index(drop=True)


SEM_LIPINSKI = asyncio.Semaphore(20)


def lipinski_rule_of_five(x: dict, smi_column: str = "smi") -> dict:
    smi = x[smi_column]
    mol = MolFromSmiles(smi)
    assert mol is not None, "invalid smi: %s" % smi
    x["num_h_bond_donors"] = Lipinski.NumHDonors(mol)
    x["lipinski_h_donor"] = x["num_h_bond_donors"] <= 5
    x["num_h_bond_acceptors"] = Lipinski.NumHAcceptors(mol)
    x["lipinski_h_acceptor"] = x["num_h_bond_acceptors"] <= 10
    x["molecular_weight"] = ExactMolWt(mol)
    x["lipinski_mwt"] = x["molecular_weight"] <= 500
    x["log_p"] = MolLogP(mol)
    x["lipinski_log_p"] = x["log_p"] <= 5
    x["lipinski"] = all(v for k, v in x.items() if k.startswith("lipinski"))
    return x


if __name__ == "__main__":
    build_probe_dataset()
