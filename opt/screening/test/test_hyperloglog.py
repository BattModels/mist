import random
import hashlib

import pytest
from lightning import Fabric
from src.hyperloglog import HyperLogLogSet


def test_hyperloglog_insertion():
    # Test basic insertion functionality
    hll = HyperLogLogSet()
    items = ["apple", "banana", "orange", "apple", "grape", "banana"]
    for item in items:
        hll.insert(item)

    # Estimate cardinality
    estimated_cardinality = len(hll)
    assert (
        estimated_cardinality >= 3
    ), f"Expected at least 3 distinct items, got {estimated_cardinality}"


def test_hyperloglog_merge():
    # Test merge functionality
    hll1 = HyperLogLogSet()
    hll2 = HyperLogLogSet()

    items1 = ["apple", "banana", "orange"]
    items2 = ["grape", "banana", "kiwi"]

    for item in items1:
        hll1.insert(item)
    for item in items2:
        hll2.insert(item)

    # Merge hll2 into hll1
    hll1.merge(hll2)

    # Expected cardinality should be at least 5 (since there are no duplicates in the union)
    estimated_cardinality = len(hll1)
    assert (
        estimated_cardinality >= 5
    ), f"Expected at least 5 distinct items after merge, got {estimated_cardinality}"


def test_hyperloglog_reduce_single():
    # Test reduce functionality with single device
    hll1 = HyperLogLogSet()
    items = ["apple", "banana", "orange", "apple", "grape"]
    for item in items:
        hll1.insert(item)

    # Simulate reduce (no actual distributed computing here)
    reduced_hll = hll1.reduce(Fabric())
    estimated_cardinality = len(reduced_hll)
    assert (
        estimated_cardinality == 4
    ), f"Expected 4 distinct items, got {estimated_cardinality}"


@pytest.mark.parametrize(
    "b,hash",
    [
        (8, hashlib.sha1),
        (12, hashlib.sha1),
        (16, hashlib.sha256),
        (12, hashlib.blake2b),
    ],
)
def test_hyperloglog_cardinality(b, hash):
    # Test cardinality estimation accuracy
    hll = HyperLogLogSet(b=b, hash=hash)

    # Insert a range of items
    num_items = 10_000
    items = [str(random.randint(1, num_items)) for _ in range(num_items)]

    for item in items:
        hll.insert(item)

    # Estimate the cardinality
    estimated_cardinality = len(hll)

    # We expect the estimate to be close to the true number of unique items
    assert (
        estimated_cardinality > 0
    ), f"Estimated cardinality is {estimated_cardinality}"
    assert (
        abs(estimated_cardinality - len(set(items))) <= len(items) * 0.05
    ), f"Cardinality estimate {estimated_cardinality} is too far from actual {len(set(items))}"


@pytest.mark.parametrize("b", [4, 8, 12, 16])
def test_hyperloglog_different_precision(b):
    # Test with different precisions (b values)
    hll = HyperLogLogSet(b)
    items = ["apple", "banana", "orange", "apple", "grape", "banana"]
    for item in items:
        hll.insert(item)

    # Cardinality should be at least 3
    estimated_cardinality = len(hll)
    assert (
        estimated_cardinality >= 3
    ), f"Expected at least 3 distinct items, got {estimated_cardinality}"
