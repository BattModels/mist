"""Application dependencies carry released or selected model directories."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HEADS = {"bp", "mp", "odour", "pka", "etn", "kt", "dn", "fp"}
RELEASED = {
    (case["base"], case.get("dataset", case.get("canonical_task")))
    for case in yaml.safe_load((ROOT / "opt/benchmarks/registry.yaml").read_text())[
        "cases"
    ]
    if case.get("release") is not None
    and case.get("target_subset") is None
    and case.get("variant") is None
}
pytestmark = pytest.mark.skipif(
    shutil.which("pomset") is None, reason="Pomset CLI required"
)


@pytest.fixture(scope="module")
def graph(inspect_workflow):
    tasks = inspect_workflow()
    return {item["id"]: item for item in tasks}


def model(producer):
    return {"producer": producer, "key": {"Named": "model"}}


def ancestors(graph, name):
    found = set()
    pending = list(graph[name]["dependencies"])
    while pending:
        parent = pending.pop()
        if parent not in found:
            found.add(parent)
            pending.extend(graph[parent]["dependencies"])
    return found


def test_screening_setup_tracks_julia_dependency_changes(copy_workflow_definitions, tmp_path):
    copy_workflow_definitions(tmp_path)

    def fingerprint():
        output = subprocess.check_output(
            ["pomset", "task", "show", "screening-julia-environment", "--json"],
            cwd=tmp_path, text=True, timeout=30,
        )
        return json.loads(output)["logical"]["execution_fingerprint"]

    original = fingerprint()
    project = tmp_path / "opt/screening/plots/Project.toml"
    project.write_text(project.read_text().replace("[compat]\n", '[compat]\nArgParse = "1"\n'))
    assert fingerprint() != original


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize(
    ("consumer", "heads"),
    [
        ("design-{base}-hydrocarbons", {"bp", "mp", "fp"}),
        ("design-{base}-electrolytes", {"bp", "mp", "fp", "dn", "kt", "pka"}),
        ("design-{base}-olfaction", {"odour"}),
        ("screening-{base}-electrolyte", {"bp", "mp", "qm9"}),
        ("screening-{base}-electrolyte-config", {"bp", "mp", "qm9"}),
        ("screening-{base}-electrolyte-plots", {"bp", "mp", "qm9", "odour"}),
        (
            "screening-{base}-lithium-air",
            {"bp", "mp", "pka", "etn", "kt", "qm9"},
        ),
        (
            "screening-{base}-lithium-air-config",
            {"bp", "mp", "pka", "etn", "kt", "qm9"},
        ),
        (
            "screening-{base}-lithium-air-plots",
            {"bp", "mp", "pka", "etn", "kt", "qm9"},
        ),
    ],
)
def test_applications_use_released_models_but_wait_for_trained_models(
    base, consumer, heads, graph
):
    consumer_task = graph[consumer.format(base=base)]
    dependencies = consumer_task["dependencies"]
    for head in heads:
        release = f"benchmark-stage-release-{base}-{head}"
        evaluated = f"benchmark-{base}-{head}-test"
        published = (base, head) in RELEASED
        assert (release in dependencies) is published
        assert (evaluated in dependencies) is not published
        assert model(release if published else evaluated) in consumer_task["inputs"]


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
def test_all_application_tests_still_feed_the_benchmark_report(base, graph):
    report = graph["validation-benchmark-report"]
    for head in HEADS:
        name = f"benchmark-{base}-{head}-test"
        assert name in report["dependencies"]
        assert {"producer": name, "key": {"Named": "metrics"}} in report["inputs"]
        evaluation = graph[name]
        assert (
            f"benchmark-{head}-materialize-application-split"
            in evaluation["dependencies"]
        )
        if (base, head) in RELEASED:
            assert (
                model(f"benchmark-stage-release-{base}-{head}") in evaluation["inputs"]
            )
        else:
            arguments = evaluation["cmd"]
            assert {"Literal": "--cells"} in arguments
            assert {"Literal": "--models"} in arguments
            model_inputs = [
                artifact
                for artifact in evaluation["inputs"]
                if artifact["key"].get("ArrayElement", {}).get("name") == "model"
            ]
            cell_inputs = [
                artifact
                for artifact in evaluation["inputs"]
                if artifact["key"].get("ArrayElement", {}).get("name") == "cell"
            ]
            assert len(model_inputs) == len(cell_inputs) > 0
        assert model(name) in [output["id"] for output in evaluation["outputs"]]


@pytest.mark.parametrize("head", ["dn", "fp"])
@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
def test_missing_training_data_only_blocks_new_heads_and_evaluation(base, head, graph):
    dataset = f"benchmark-{head}-materialize-application-split"
    published = (base, head) in RELEASED
    assert (dataset in ancestors(graph, f"design-{base}-electrolytes")) is not published
    assert dataset in ancestors(graph, f"benchmark-{base}-{head}-test")


def test_model_stages_expose_directories_and_no_contract_tasks_remain(graph):
    assert not any(name.startswith("benchmark-contract-") for name in graph)
    staged = [
        item for name, item in graph.items() if name.startswith("benchmark-stage-")
    ]
    assert staged
    for item in staged:
        assert [output["id"] for output in item["outputs"]] == [model(item["id"])]
        assert not item["outputs"][0]["path"].endswith(".json")
        assert {"Literal": "--output"} in item["cmd"]
