from copy import deepcopy
from itertools import islice

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator, train_test_split
from datasets import Dataset, DatasetDict


class EntityHoldoutSplitter(BaseCrossValidator):
    def __init__(
        self,
        entity_cols: list[str],
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state=42,
    ):
        self.entity_cols = entity_cols
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self._last_held_out = None

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.random_state)
        entities = pd.unique(X[self.entity_cols].values.ravel())
        n_holdout = max(1, int(self.test_size * len(entities)))
        for i in range(self.n_splits):
            held_out = rng.choice(entities, size=n_holdout, replace=False)
            self._last_held_out = held_out

            def hold_out_row(row):
                for col in self.entity_cols:
                    if row[col] in held_out:
                        return True
                return False

            test_mask = X.apply(hold_out_row, axis=1)
            train_idx = X.index[~test_mask].to_numpy()
            test_idx = X.index[test_mask].to_numpy()
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StrictEntityHoldoutSplitter(BaseCrossValidator):
    def __init__(
        self,
        entity_cols: list[str],
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state=42,
        max_retries=25,
    ):
        self.entity_cols = entity_cols
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_retries = max_retries
        self._last_held_out = None

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.random_state)
        entities = pd.unique(X[self.entity_cols].values.ravel())
        n_holdout = max(1, int(self.test_size * len(entities)))
        target_test_size = int(self.test_size * len(X))

        for _ in range(self.n_splits):
            best_test_idx = None
            best_held_out = None

            for _ in range(self.max_retries):
                held_out = set(rng.choice(entities, size=n_holdout, replace=False))

                def is_test(row):
                    return all(row[col] in held_out for col in self.entity_cols)

                def is_train(row):
                    return all(row[col] not in held_out for col in self.entity_cols)

                test_mask = X.apply(is_test, axis=1)
                train_mask = X.apply(is_train, axis=1)

                test_idx = X.index[test_mask].to_numpy()
                train_idx = X.index[train_mask].to_numpy()

                if best_test_idx is None or len(test_idx) > len(best_test_idx):
                    best_test_idx = test_idx
                    best_train_idx = train_idx
                    best_held_out = held_out

                if len(test_idx) >= target_test_size:
                    break  # Acceptable split

            self._last_held_out = best_held_out
            yield best_train_idx, best_test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def apply_splitter(ds: Dataset | pd.DataFrame, splitter, split: int = 0, groups=None):
    df = ds.to_pandas()

    if groups is not None:
        groups = [tuple(x) for x in df[groups].values]
        groups, _ = pd.factorize(groups)

    splits = splitter.split(df, groups=groups)
    train_idx, val_idx = list(islice(splits, split + 1))[-1]

    if hasattr(splitter, "_last_held_out"):
        print("Held out (val/test):", splitter._last_held_out)

    # Subset the validation dataframe
    val_df = df.loc[val_idx].reset_index(drop=True)

    # Run splitter again on val_df
    val_groups = None
    if groups is not None:
        val_group_vals = [tuple(x) for x in val_df[groups].values]
        val_group_ids, _ = pd.factorize(val_group_vals)
        val_groups = val_group_ids

    # Update splitter for a 50/50 split
    splitter = deepcopy(splitter)
    splitter.test_size = 0.5
    second_splits = splitter.split(val_df, groups=val_groups)
    val_inner_idx, test_inner_idx = next(second_splits)

    if hasattr(splitter, "_last_held_out"):
        print("Held out (test):", splitter._last_held_out)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df.loc[train_idx], preserve_index=False),
            "validation": Dataset.from_pandas(
                val_df.loc[val_inner_idx], preserve_index=False
            ),
            "test": Dataset.from_pandas(
                val_df.loc[test_inner_idx], preserve_index=False
            ),
        }
    )
