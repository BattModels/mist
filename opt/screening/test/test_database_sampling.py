"""Dense fragment artifacts retain the existing repeated-batch sampling rule."""

import sqlite3

from src.dataloader import DatabaseFragmentDataset


def database(path, size):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fragments (fragment TEXT)")
        connection.executemany(
            "INSERT INTO fragments VALUES (?)", [(str(i),) for i in range(size)]
        )
    return path


def test_large_samples_keep_independent_batches(tmp_path, monkeypatch):
    path = database(tmp_path / "fragments.sqlite", 10_001)
    batches = []

    def sample(population, k):
        batches.append(k)
        return list(population)[:k]

    monkeypatch.setattr("src.dataloader.random.sample", sample)
    dataset = DatabaseFragmentDataset(None, None, path, n_fragments=10_002)
    fragments = list(dataset.fragments())
    assert batches == [10_000, 2]
    assert len(fragments) == 10_002
    assert len(set(fragments)) == 10_000
    assert fragments.count("0") == fragments.count("1") == 2


def test_fraction_limit_accepts_dense_rowids(tmp_path):
    path = database(tmp_path / "fragments.sqlite", 100)
    dataset = DatabaseFragmentDataset(
        None, None, path, n_fragments=10, limit_db_fragments=0.1
    )
    assert set(dataset.fragments()) == {str(i) for i in range(10)}
