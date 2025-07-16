import pandas as pd
import pytest
from electrolyte_fm.data_modules.splits import (
    EntityHoldoutSplitter,
    StrictEntityHoldoutSplitter,
)


@pytest.fixture
def toy_df():
    data = {
        "x": ["A", "B", "C", "A", "B", "C", "D", "E", "F", "G"],
        "y": ["B", "C", "A", "C", "A", "B", "E", "F", "G", "D"],
        "temperature": [300, 310, 320, 300, 310, 320, 330, 340, 350, 360],
        "value": [1.0] * 10,
    }
    return pd.DataFrame(data)


def test_number_of_splits(toy_df):
    splitter = EntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=3, test_size=0.2, random_state=1
    )
    splits = list(splitter.split(toy_df))
    assert len(splits) == 3


def test_number_of_splits_strict(toy_df):
    splitter = StrictEntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=3, test_size=0.2, random_state=1
    )
    splits = list(splitter.split(toy_df))
    assert len(splits) == 3


def test_entity_exclusion(toy_df):
    splitter = EntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=1, test_size=0.2, random_state=1
    )
    train_idx, test_idx = next(splitter.split(toy_df))
    train_df = toy_df.iloc[train_idx]
    test_df = toy_df.iloc[test_idx]

    held_out_entities = set(splitter._last_held_out)
    train_entities = set(train_df["x"]).union(set(train_df["y"]))
    assert len(held_out_entities.intersection(train_entities)) == 0


def test_entity_exclusion_strict(toy_df):
    splitter = StrictEntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=1, test_size=0.2, random_state=1
    )
    train_idx, test_idx = next(splitter.split(toy_df))
    train_df = toy_df.iloc[train_idx]
    test_df = toy_df.iloc[test_idx]
    train_entities = set(train_df["x"]).union(set(train_df["y"]))
    test_entities = set(test_df["x"]).union(set(test_df["y"]))
    assert len(train_entities.intersection(test_entities)) == 0


def test_test_rows_have_held_out_entity(toy_df):
    splitter = EntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=1, test_size=0.3, random_state=42
    )
    train_idx, test_idx = next(splitter.split(toy_df))
    test_df = toy_df.iloc[test_idx]

    held_out_entities = pd.unique(test_df[["x", "y"]].values.ravel())
    for _, row in test_df.iterrows():
        assert row["x"] in held_out_entities or row["y"] in held_out_entities


def test_disjoint_train_test(toy_df):
    splitter = EntityHoldoutSplitter(
        entity_cols=["x", "y"], n_splits=1, test_size=0.3, random_state=42
    )
    for train_idx, test_idx in splitter.split(toy_df):
        intersection = set(train_idx) & set(test_idx)
        assert len(intersection) == 0
