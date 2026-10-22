"""Both model families have matching plots and distinct model dependencies."""

import shutil
from collections import Counter
from pathlib import PurePosixPath

import pytest

pytestmark = pytest.mark.skipif(shutil.which("pomset") is None, reason="Pomset CLI required")
BASES = ("mist-28m", "mist-1p8b")


@pytest.fixture(scope="module")
def graph(inspect_workflow):
    return {task["id"]: task for task in inspect_workflow()}


@pytest.mark.parametrize("area", [
    "interpretability", "FeatureMiner", "isotopes", "mixtures",
    "ionic-conductivity", "olfaction",
])
def test_both_models_have_the_same_figure_inventory(graph, area):
    inventories = []
    for base in BASES:
        prefix = f"//results/{area}/{base}/"
        outputs = [
            PurePosixPath(output["path"]).name.replace(base, "MODEL")
            for task in graph.values() if "figures" in task["tags"]
            for output in task["outputs"]
            if output["path"].startswith(prefix) and output["path"].endswith(".pdf")
        ]
        assert outputs, (area, base)
        inventories.append(Counter(outputs))
    assert inventories[0] == inventories[1]


@pytest.mark.parametrize("consumer,producer", [
    ("embeddings-mist-1p8b-extract-tokens-pretrained", "benchmark-stage-mist-1p8b"),
    ("embeddings-mist-1p8b-extract-tokens-toxcast", "benchmark-mist-1p8b-toxcast-test"),
    ("embeddings-mist-28m-encode-aromaticity", "benchmark-stage-mist-28m"),
    ("isotopes-mist-1p8b-embeddings", "benchmark-mist-1p8b-isotopes-test"),
    ("mixtures-mist-1p8b-paper-parity", "benchmark-mist-1p8b-mixtures-random"),
    ("ionic-mist-1p8b-delta-ea", "benchmark-mist-1p8b-ionic-conductivity-fit"),
    ("olfaction-mist-1p8b-all-descriptors-predictions", "benchmark-mist-1p8b-odour-test"),
])
def test_new_analyses_use_the_matching_model(graph, consumer, producer):
    assert {"producer": producer, "key": {"Named": "model"}} in graph[consumer]["inputs"]


@pytest.mark.parametrize("base", BASES)
def test_experimental_plots_use_the_matching_strict_fold(graph, base):
    for fold in range(5):
        name = "paper-experimental" if fold == 0 else f"strict-fold-{fold}-experimental"
        prefix = "mixtures-" if base == "mist-28m" else f"mixtures-{base}-"
        assert {
            "producer": f"benchmark-{base}-mixtures-strict-{fold}",
            "key": {"Named": "model"},
        } in graph[prefix + name]["inputs"]


def test_output_artifacts_are_unique(graph):
    paths = [output["path"] for task in graph.values() for output in task["outputs"]]
    assert len(paths) == len(set(paths))
