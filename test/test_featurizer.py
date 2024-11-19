import re
from random import choice
from itertools import chain

import pytest
import torch

from electrolyte_fm.data_modules.feature_tagger import (
    ELEMENT_FEATURES,
    ELEMENT_SYMBOLS,
    REGEX_FEATURES,
    RegexFeature,
    ElementFeature,
    SMARTSFeature,
)

REGEX_TESTS = [
    {
        "feature": "chiral_tags",
        "positive": ["@SP1", "@OH32", "@TB2", "@", "@@"],
        "negative": ["C", "O", "c", "[Rb]"],
    },
    {
        "feature": "bracked_atom",
        "positive": ["[C]", "[C-]", "[C+]", "[C@]", "[C@@]", "[C@H]", "[C@H-]"],
        "negative": ["C", "O", "c", "F"],
    },
    {
        "feature": "charged_atom",
        "positive": ["[C+]", "[C-]", "[C++]", "[Rb@OH2+3]"],
        "negative": ["C", "O", "c", "[Rb@OH2]"],
    },
    {
        "feature": "chiral_center",
        "positive": ["[C@]", "[C@@]", "[C@H]", "[C@H-]"],
        "negative": ["C", "O", "c", "[Rb]"],
    },
    {
        "feature": "aromatic_bracket_atom",
        "positive": ["[te+2]", "[b]", "[c@@]"],
        "negative": ["C", "O", "[Rb]", "[Sn]", "[Cn]"],
    },
]

SMARTS_TESTS = [
    {
        "feature": "ketone",
        "positive": ["CC(=O)C", "C[C@@H]1CCCCCCCCCCCCC(=O)C1"],
        "negative": ["C=O", "OCC"],
    },
    {
        "feature": "aldehyde",
        "positive": ["C=O", "O=CC", "CC=O"],
        "negative": ["CC(=O)C", "OCC(O)CO"],
    },
    {
        "feature": "carboxylic_acid",
        "positive": ["O=CO", "CC(=O)O", "C(=O)O", "O=C(O)C"],
        "negative": ["CC(=O)C", "OCC"],
    },
    {
        "feature": "amid",
        "positive": ["O=CN", "CC(=O)N", "C(=O)NC", "O=C(N)C"],
        "negative": ["CC(=O)C", "OCC"],
    },
    {
        "feature": "hydroxyl",
        "positive": ["CO", "CCO", "C(O)C"],
        "negative": ["CC(=O)C", "C=O"],
    },
    {
        "feature": "phenol",
        "positive": [
            "c1ccc(cc1)O",
            "Oc1ccccc1",
            "Oc0ccccc0Cc0cc(C1)c(O)c(c0)Cc0c(O)ccc(c0)Cc0ccc(O)c(c0)Cc0c(O)ccc(c0)Cc0c(O)ccc(c0)Cc0c(O)c(C2)cc(c0)Cc0c(O)ccc(c0)Cc(c0O)cc2cc0Cc0cc(Cc2ccc(O)cc2)c(O)c(c0)Cc0c(O)ccc(c0)C1",
        ],
        "negative": [
            "CC(=O)C",
            "C=O",
            "c1ccccc1-c2ccccc2",
            "c1ccc(cc1)C[C@@H](C(=O)O)N",
        ],
    },
    {
        "feature": "rotatable_bond",
        "positive": [
            "CC-CC",
            "c1ccccc1-c2ccccc2",
        ],
        "negative": ["CC(=O)C", "C=O", "c1ccccc1"],
    },
]


@pytest.mark.parametrize(
    "feature,negatives",
    ((x["feature"], x["negative"]) for x in REGEX_TESTS if "negative" in x),
)
def test_regex_negatives(feature: str, negatives: list[str]):
    f = re.compile(REGEX_FEATURES[feature])
    for neg in negatives:
        assert f.match(neg) is None, f"{feature} should not match {neg}"


@pytest.mark.parametrize(
    "feature,positives",
    ((x["feature"], x["positive"]) for x in REGEX_TESTS if "positive" in x),
)
def test_regex_negatives(feature: str, positives: list[str]):
    f = re.compile(REGEX_FEATURES[feature])
    for pos in positives:
        assert f.match(pos) is not None, f"{feature} should match {pos}"


def generate_examples():
    for x in REGEX_TESTS:
        yield RegexFeature, x["feature"], x["positive"], x["negative"]

    elements = set(ELEMENT_SYMBOLS)
    for name, positive in ELEMENT_FEATURES.items():
        pos_examples = [f"[{e}]" for e in positive]
        neg_examples = [f"[{e}]" for e in elements - set(positive)]
        yield ElementFeature, name, pos_examples, neg_examples

    for x in SMARTS_TESTS:
        yield SMARTSFeature, x["feature"], x["positive"], x["negative"]


@pytest.mark.parametrize(
    "cls,feature,positive,negative",
    generate_examples(),
)
def test_positive_feature(cls, feature, positive, negative):
    f = cls.from_named(feature)
    for pos in positive:
        active = f.featurize(pos)
        assert active.any(), "{} should match {}: {}".format(feature, pos, active)


@pytest.mark.parametrize(
    "cls,feature,positive,negative",
    generate_examples(),
)
def test_negative_feature(cls, feature, positive, negative):
    f = cls.from_named(feature)
    for neg in negative:
        active = f.featurize(neg)
        assert not active.any(), "{} should not match {}: {}".format(
            feature, neg, active
        )


@pytest.mark.parametrize(
    "cls,feature,positive,negative",
    [e for e in generate_examples() if e[0] != SMARTSFeature],
)
def test_alignment(cls, feature, positive, negative):
    f = cls.from_named(feature)
    pos = choice(positive)
    neg = choice(negative)
    active = f.featurize(pos)
    inactive = f.featurize(neg)
    assert active.any() and not inactive.any()
    check_active(pos + neg, torch.cat([active, inactive]), f.featurize(pos + neg))
    check_active(pos + pos, torch.cat([active, active]), f.featurize(pos + pos))
    check_active(
        neg + pos + neg,
        torch.cat([inactive, active, inactive]),
        f.featurize(neg + pos + neg),
    )


def check_active(smi, expected, actual):
    print(f"smi: {smi}")
    print(f"expected: {expected}")
    print(f"actual:   {actual}")
    assert isinstance(actual, torch.BoolTensor)
    assert isinstance(expected, torch.BoolTensor)
    assert all(actual == expected)
