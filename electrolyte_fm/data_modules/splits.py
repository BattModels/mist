import logging
from collections import Counter
from collections.abc import Sequence, Iterator

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class EntityHoldoutSplitter(BaseCrossValidator):
    """
    A cross-validator that holds out based on entity IDs, with an option for strict
    vs non-strict membership, preserves group integrity, and approximately preserves
    the distribution of y. Can optionally emit verbose statistics about each split.
    """

    def __init__(
        self,
        entity_cols: list[str],
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state: int | None = 42,
        strict: bool = False,
        distrib_tolerance: float = 0.1,
        verbose: bool = False,
        shuffle: bool = False,
    ):
        self.entity_cols = entity_cols
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.strict = strict
        self.distrib_tolerance = distrib_tolerance
        self.verbose = verbose
        self.shuffle = shuffle
        self._last_held_out: set | None = None  # introspection

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: np.ndarray | None = None,
        groups: Sequence[int] | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None = None,
        groups: Sequence[int] | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        X = X.reset_index(drop=True)
        n_samples = len(X)

        # set up groups array
        if groups is None:
            groups_arr = np.arange(n_samples)
        else:
            groups_arr = np.array(groups)

        # check that entity_cols are constant within each group (order-insensitive); if not, show differing rows
        for g in np.unique(groups_arr):
            subset_df = X.loc[groups_arr == g, self.entity_cols]
            first_set = set(subset_df.iloc[0].values)
            diffs = []
            for idx, row in subset_df.iterrows():
                row_set = set(row.values)
                if row_set != first_set:
                    diffs.append((int(idx), tuple(row.values)))
            if diffs:
                raise AssertionError(
                    f"Entity columns vary within group {g}. canonical_set={first_set}; differing rows={diffs}"
                )  # enforce constant membership

        # compute overall y-distribution
        if y is not None:
            y_arr = np.array(y)
            overall_counts = Counter(y_arr)
            overall_frac = {k: v / n_samples for k, v in overall_counts.items()}
        else:
            y_arr = None

        all_entities = np.unique(X[self.entity_cols].values.ravel())
        rng = np.random.default_rng(self.random_state)

        # Precompute group -> entity set (constant per group by assertion)
        unique_groups = np.unique(groups_arr)
        group_entity_sets = {}
        for g in unique_groups:
            mask = groups_arr == g
            row = X.loc[mask, self.entity_cols].iloc[0]
            group_entity_sets[g] = set(row.values)

        for split_i in range(self.n_splits):
            # attempt to satisfy distribution
            logging.debug("Starting construction of split %d", split_i)
            for _ in range(10):
                perm = rng.permutation(all_entities)
                held_out = set()
                test_idx = np.array([], dtype=int)

                # accumulate until enough, using precomputed sets
                for ent in perm:
                    held_out.add(ent)
                    test_groups = []
                    if self.strict:
                        # group is test if its entire entity set is subset of held_out
                        for g in unique_groups:
                            if group_entity_sets[g].issubset(held_out):
                                test_groups.append(g)
                    else:
                        # group is test if any overlap
                        for g in unique_groups:
                            if group_entity_sets[g] & held_out:
                                test_groups.append(g)
                    test_idx = np.where(np.isin(groups_arr, test_groups))[0]
                    logging.debug("test fraction: %f", len(test_idx) / n_samples)
                    if len(test_idx) / n_samples >= self.test_size:
                        break

                # check y-distribution
                if y_arr is None:
                    break
                test_counts = Counter(y_arr[test_idx])
                test_frac = {
                    k: test_counts.get(k, 0) / max(1, len(test_idx))
                    for k in overall_frac
                }
                distrib_error = {
                    k: abs(test_frac[k] - overall_frac[k]) for k in overall_frac.keys()
                }
                logging.debug("Current distrib_error: %s", distrib_error)
                if all(v <= self.distrib_tolerance for v in distrib_error.values()):
                    break

            self._last_held_out = held_out

            # generate train indices
            if self.strict:
                # no row with any held-out entity
                mask = (
                    X[self.entity_cols]
                    .apply(lambda r: set(r).isdisjoint(held_out), axis=1)
                    .to_numpy()
                )
                train_idx = np.where(mask)[0]
            else:
                train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

            # compute and optionally log stats
            stats = self._compute_split_stats(
                X,
                y_arr,
                train_idx,
                test_idx,
                overall_frac if y_arr is not None else None,
            )
            if self.verbose:
                logger.info(f"Split {split_i}: {stats}")

            if self.shuffle:
                rng.shuffle(train_idx)
                rng.shuffle(test_idx)

            yield train_idx, test_idx

    def _compute_split_stats(
        self,
        X: pd.DataFrame,
        y_arr: np.ndarray | None,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        overall_frac: dict | None = None,
    ) -> dict:
        """Return statistics about the last split."""
        stat: dict = {}
        stat["train_size"] = len(train_idx)
        stat["test_size"] = len(test_idx)
        stat["n_entities_held_out"] = (
            len(self._last_held_out) if self._last_held_out is not None else 0
        )

        def frac_counts(idxs: np.ndarray) -> dict:
            if y_arr is None:
                return {}
            counts = Counter(y_arr[idxs])
            total = len(idxs) if len(idxs) > 0 else 1
            return {
                k: counts.get(k, 0) / total
                for k in (overall_frac.keys() if overall_frac else counts.keys())
            }

        if y_arr is not None and overall_frac is not None:
            stat["overall_y_frac"] = overall_frac
            stat["train_y_frac"] = frac_counts(train_idx)
            stat["test_y_frac"] = frac_counts(test_idx)
        # entities present in training set (excluding held-out)
        train_entities = np.unique(X.loc[train_idx, self.entity_cols].values.ravel())
        stat["n_entities_in_train"] = len(train_entities)
        return stat


def apply_splitter(df: Dataset | pd.DataFrame, splitter, **kwargs):
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    for train, val in splitter.split(df, **kwargs):
        yield (df.iloc[train], df.iloc[val])


def stratified_mixture_sparsity_split(
    ds: Dataset,
    splitter,
    target_columns: list[str],
    mixture_id: str = "mixture_id",
    **kwargs,
):
    df = ds.to_pandas()

    # Stratify by target_columns sparsity
    def compute_sparsity(gdf):
        bits = []
        for col in target_columns:
            has_data = False
            for c in df.columns.tolist():
                if not (c.startswith(col) or c.startswith(f"excess {col}")):
                    continue
                if gdf[c].notna().any():
                    has_data = True
                    break
            bits.append("1" if has_data else "0")
        return pd.Series({"sparsity": "".join(bits)})

    # Group by mixture_id and computed sparsity
    sparsity_df = (
        df.groupby(mixture_id)
        .apply(compute_sparsity, include_groups=False)
        .reset_index()
    )
    sparsity_df = merge_small_sparsity_groups_by_hamming(
        sparsity_df,
        "sparsity",
        min_size=sparsity_df.shape[0] // 20,
    )
    df = df.merge(sparsity_df, on="mixture_id", how="left")
    sparsity = df["sparsity"]
    groups = df["mixture_id"]
    df.drop("sparsity", axis=1, inplace=True)

    for train, val in apply_splitter(df, splitter, y=sparsity, groups=groups):
        yield DatasetDict(
            {
                "train": Dataset.from_pandas(train),
                "validation": Dataset.from_pandas(val),
            }
        )


def merge_small_sparsity_groups_by_hamming(
    df: pd.DataFrame,
    group_col: str,
    min_size: int = 10,
) -> pd.DataFrame:
    counts = df[group_col].value_counts()

    small_groups = counts[counts < min_size].index.tolist()
    large_groups = counts[counts >= min_size].index.tolist()

    # Mapping from small group to nearest large group
    merge_map = {}

    def hamming_dist(s1: str, s2: str) -> int:
        return sum(a != b for a, b in zip(s1, s2))

    for small in small_groups:
        dists = [
            (hamming_dist(small, large), -counts[large], large)
            for large in large_groups
        ]
        dists.sort()  # Sort by (distance, -count, lex)
        nearest_large = dists[0][2]
        merge_map[small] = nearest_large

    # Apply the mapping
    df["sparsity"] = df["sparsity"].replace(merge_map)

    return df
