import pytest
from electrolyte_fm.data_modules.feature_tagger import REGEX_FEATURES


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
    {"feature": "aromatic_atom", 
     "positive": ["[te+2]", "b", "c", "o", "p", "se", "as"],
     "negative": ["C", "O", "[Rb]"],
    },
]


@pytest.mark.parametrize(
    "feature,negatives",
    ((x["feature"], x["negative"]) for x in REGEX_TESTS if "negative" in x),
)
def test_regex_negatives(feature: str, negatives: list[str]):
    for neg in negatives:
        assert REGEX_FEATURES[feature].match(neg) is None, f"{feature} should not match {neg}"


@pytest.mark.parametrize(
    "feature,positives",
    ((x["feature"], x["positive"]) for x in REGEX_TESTS if "positive" in x),
)
def test_regex_negatives(feature: str, positives: list[str]):
    for pos in positives:
        assert REGEX_FEATURES[feature].match(pos) is not None, f"{feature} should match {pos}"
