import logging
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from electrolyte_fm.data_modules.splits import EntityHoldoutSplitter


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 500
    # define possible entities
    entities = [f"E{i}" for i in range(10)]
    # assign each row to one of 10 groups
    groups = rng.integers(0, 10, size=n)
    # for each group, choose one fixed entity-pair
    unique_grps = np.unique(groups)
    ent_map = {
        g: tuple(rng.choice(entities, size=2, replace=False)) for g in unique_grps
    }
    A = [ent_map[g][0] for g in groups]
    B = [ent_map[g][1] for g in groups]
    X = pd.DataFrame({"A": A, "B": B})
    # y as constant bitstring per group
    y = np.empty(n, dtype=object)
    for g in unique_grps:
        bitstr = "".join(str(int(x)) for x in rng.integers(0, 2, size=3))
        y[groups == g] = bitstr
    return X, y, groups


def test_basic_split_size(synthetic_data):
    X, y, groups = synthetic_data
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.3,
        n_splits=1,
        random_state=42,
        strict=False,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    # sizes
    assert abs(len(test_idx) - 0.3 * len(X)) / len(X) <= 0.05  # within 5% tolerance
    assert len(train_idx) + len(test_idx) == len(X)


def test_strict_holdout(synthetic_data):
    X, y, groups = synthetic_data
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.4,
        n_splits=1,
        random_state=1,
        strict=True,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    held_out = splitter._last_held_out
    # Every test row has *both* its entities in held_out
    for i in test_idx:
        assert set(X.loc[i, ["A", "B"]]).issubset(held_out)
    # Every train row has *neither* entity in held_out
    for i in train_idx:
        assert set(X.loc[i, ["A", "B"]]).isdisjoint(held_out)


def test_non_strict_holdout(synthetic_data):
    X, y, groups = synthetic_data
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.4,
        n_splits=1,
        random_state=1,
        strict=False,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    held_out = splitter._last_held_out
    # Every test row has *at least one* entity in held_out
    for i in test_idx:
        assert len(set(X.loc[i, ["A", "B"]]) & held_out) >= 1
    # Train rows have no overlap
    for i in train_idx:
        assert set(X.loc[i, ["A", "B"]]).isdisjoint(held_out)


def test_group_consistency(synthetic_data):
    X, y, groups = synthetic_data
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.5,
        n_splits=1,
        random_state=2,
        strict=False,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    # No group appears in both train and test
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(test_groups)


def test_target_distribution(synthetic_data):
    X, y, groups = synthetic_data
    # Check that approximate distribution of bit-strings is preserved
    overall_counts = Counter(y)
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.2,
        n_splits=1,
        random_state=3,
        strict=False,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    train_counts = Counter(y[train_idx])
    test_counts = Counter(y[test_idx])
    # For each bitstring, the fraction in train/test should be within ±15% of overall
    tolerance = 0.15
    for bitstr, overall_count in overall_counts.items():
        overall_frac = overall_count / len(y)
        test_frac = test_counts.get(bitstr, 0) / len(test_idx)
        train_frac = train_counts.get(bitstr, 0) / len(train_idx)
        assert abs(test_frac - overall_frac) < tolerance
        assert abs(train_frac - overall_frac) < tolerance


def test_multiple_splits_randomness(synthetic_data):
    X, y, groups = synthetic_data
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.3,
        n_splits=5,
        random_state=5,
        strict=False,
    )
    splits = list(splitter.split(X, y, groups))
    # Should get n_splits splits
    assert len(splits) == 5
    # Test sets should not all be identical
    test_sets = [set(test_idx) for _, test_idx in splits]
    # If they were all identical, the set of frozensets would have size 1
    assert len({frozenset(ts) for ts in test_sets}) > 1


def test_verbose_stats_and_consistency(synthetic_data, caplog):
    X, y, groups = synthetic_data
    # enable logging capture
    caplog.set_level(logging.INFO)
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.3,
        n_splits=1,
        random_state=123,
        strict=True,
        verbose=True,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    # grab stats via internal method (should match what was logged)
    y_arr = np.array(y)
    overall_counts = Counter(y_arr)
    overall_frac = {k: v / len(y_arr) for k, v in overall_counts.items()}
    stats = splitter._compute_split_stats(X, y_arr, train_idx, test_idx, overall_frac)

    # Basic presence of keys
    expected_keys = {
        "train_size",
        "test_size",
        "n_entities_held_out",
        "overall_y_frac",
        "train_y_frac",
        "test_y_frac",
        "n_entities_in_train",
    }
    assert expected_keys.issubset(set(stats.keys()))

    # Sizes add up to at most len(X) -- Strict can drop compounds
    assert stats["train_size"] + stats["test_size"] <= len(X)

    # Strict: no train row can contain any held-out entity
    held_out = splitter._last_held_out
    for i in train_idx:
        assert set(X.loc[i, ["A", "B"]]).isdisjoint(held_out)

    # Distribution fractions are between 0 and 1 and sum approximately to 1
    for frac_dict in (
        stats["train_y_frac"],
        stats["test_y_frac"],
        stats["overall_y_frac"],
    ):
        for v in frac_dict.values():
            assert 0.0 <= v <= 1.0
        total = sum(frac_dict.values())
        assert pytest.approx(1.0, rel=1e-2) == total  # allow small rounding drift

    # Number of entities in train should exclude held-out ones
    train_entities = set(np.unique(X.loc[train_idx, ["A", "B"]].values.ravel()))
    assert stats["n_entities_in_train"] == len(train_entities)
    assert held_out.isdisjoint(train_entities)

    # Check that something was logged about the split
    assert any("Split 0" in record.message for record in caplog.records)
    # Optionally: the logged dict should contain same values
    assert str(stats) in "".join(r.message for r in caplog.records)


def test_entity_cols_order_insensitivity(synthetic_data):
    # mutate synthetic_data so that within a group the two entity columns
    # are swapped in one row (same set, different order)
    X, y, groups = synthetic_data
    # pick a group and swap its entity columns in one row
    g = groups[0]
    idxs = np.where(groups == g)[0]
    if len(idxs) < 2:
        pytest.skip("need a group with >=2 rows for this test")
    i0, i1 = idxs[0], idxs[1]
    # make row i1 have reversed columns of i0
    X.loc[i1, ["A", "B"]] = X.loc[i0, ["B", "A"]].values
    splitter = EntityHoldoutSplitter(
        entity_cols=["A", "B"],
        test_size=0.2,
        n_splits=1,
        strict=True,
        random_state=0,
    )
    # Should not raise, because set-wise the entity columns are the same
    _ = next(splitter.split(X, y, groups))
