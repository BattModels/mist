"""Model and dataset consumers use artifacts rather than receipt indirection."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("pomset") is None, reason="Pomset CLI required"
)


@pytest.fixture(scope="module")
def graph(inspect_workflow):
    return {task["id"]: task for task in inspect_workflow()}


def artifact(producer, name):
    return {"producer": producer, "key": {"Named": name}}


def root_path(value, origin=""):
    if value.startswith("//"):
        return ROOT / value[2:]
    return ROOT / origin / value


def command_paths(task):
    """Resolve direct arguments and model fields passed by array items."""
    paths = []
    for arg in task["cmd"]:
        if "Path" in arg:
            paths.append(arg["Path"])
        elif arg.get("Array", {}).get("source") == "item_field":
            key = arg["Array"]["key"]
            for item in task["executor"]["Array"]["items"]:
                value = item["value"][key]
                if value["type"] == "path":
                    paths.append(value["value"])
    return {
        root_path(path["path"]["value"], path["origin"]["value"])
        for path in paths
        if path["origin"]["kind"] == path["path"]["kind"] == "static"
    }


def test_tasks_no_longer_consume_receipt_indirection(graph):
    assert not [
        (task["id"], item["producer"])
        for task in graph.values()
        for item in task["inputs"]
        if item["key"] == {"Named": "receipt"}
    ]


@pytest.mark.parametrize(
    ("consumer", "producer"),
    [
        ("benchmark-validate-canonical-tasks", "benchmark-sync-environment"),
        ("benchmark-pubchemqc-materialize", "benchmark-sync-environment"),
        ("design-julia-environment", "design-sync-environment"),
        ("design-mist-28m-partial-charges", "design-sync-environment"),
        ("feature-miner-julia-environment", "feature-miner-sync-environment"),
        ("feature-miner-materialize-data", "feature-miner-sync-environment"),
        ("screening-mist-28m-electrolyte", "screening-python-environment"),
        ("synth-access-julia-environment", "synth-access-sync-environment"),
        ("synth-access-heuristics", "synth-access-materialize-inputs"),
        ("tmqm-materialize-paper-release", "tmqm-sync-environment"),
        ("tmqm-chemberta-v3-lr-sweep", "tmqm-chemberta-v3-package-model"),
        ("tmqm-chemberta-v3-package-model", "tmqm-chemberta-v3-remote-code"),
        ("benchmark-mist-28m-pubchemqc-test", "benchmark-stage-mist-28m"),
        ("benchmark-mist-1p8b-pubchemqc-test", "benchmark-stage-mist-1p8b"),
    ],
)
def test_ordering_survives_without_unused_receipt_inputs(graph, consumer, producer):
    assert producer in graph[consumer]["dependencies"]
    assert artifact(producer, "receipt") not in graph[consumer]["inputs"]


@pytest.mark.parametrize(
    ("consumer", "producer"),
    [
        ("tmqm-mist-28m-lr-sweep", "tmqm-mist-28m-model"),
        ("tmqm-mist-28m-release-test", "tmqm-mist-28m-release-model"),
        ("benchmark-mist-28m-pubchemqc-lr-sweep-bounded", "benchmark-stage-mist-28m"),
        ("benchmark-mist-28m-bp-test", "benchmark-stage-release-mist-28m-bp"),
        ("feature-miner-mist-28m-head-probes", "tmqm-mist-28m-release-model"),
        ("feature-miner-mist-28m-head-probes", "benchmark-stage-mist-28m"),
        ("synth-access-mist-28m-surprise", "benchmark-stage-mist-28m"),
    ],
)
def test_model_consumers_use_model_artifacts(graph, consumer, producer):
    assert producer in graph[consumer]["dependencies"]
    assert artifact(producer, "model") in graph[consumer]["inputs"]
    assert artifact(producer, "receipt") not in graph[consumer]["inputs"]


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize("phase", ["lr-sweep-bounded", "test"])
def test_charge_consumers_use_the_corpus_directory_artifact(graph, base, phase):
    producer = "benchmark-pubchemqc-materialize"
    task = graph[f"benchmark-{base}-pubchemqc-{phase}"]
    assert artifact(producer, "dataset") in task["inputs"]
    assert artifact(producer, "receipt") not in task["inputs"]
    assert artifact(producer, "provenance") not in task["inputs"]
    output = next(item for item in graph[producer]["outputs"] if item["id"] == artifact(producer, "dataset"))
    assert root_path(output["path"]) in command_paths(task)


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize("campaign,gpus", [("electrolyte", 8), ("lithium-air", 4)])
def test_screening_config_and_direct_command_keep_model_artifact_edges(
    graph, tmp_path, base, campaign, gpus
):
    name = f"screening-{base}-{campaign}"
    configured = graph[name + "-config"]
    screened = graph[name]
    output = tmp_path / "fresh/campaign/input-config.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            configured["cmd"][2]["Literal"],
            configured["cmd"][3]["Literal"],
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    config = json.loads(output.read_text())
    model_inputs = [
        item for item in configured["inputs"] if item["key"] == {"Named": "model"}
    ]
    assert model_inputs
    assert all(item in screened["inputs"] for item in model_inputs)
    model_paths = {
        str(root_path(output["path"]))
        for item in model_inputs
        for output in graph[item["producer"]]["outputs"]
        if output["id"] == item
    }
    working_directory = root_path(screened["working_directory"]["value"])
    assert {
        str((working_directory / critic["model_path"]).resolve())
        for critic in config["critics"]
    } == model_paths
    assert all(not path.endswith("receipt.json") for path in model_paths)
    settings = dict(config["generation"])
    assert (working_directory / settings.pop("db_path")).resolve() == (
        ROOT / "results/screening/zinc-fragments.sqlite"
    )
    assert (working_directory / settings.pop("reference_fragments")).resolve() == (
        ROOT / f"results/screening/references/{campaign}.frag"
    )
    assert settings == {
        "batch_size": 256 if base == "mist-1p8b" and campaign == "lithium-air" else 768,
        "n_fragments": 1000,
        "epoch_size": 1200,
        "limit_db_fragments": None,
        "limit_ref_fragments": None,
    }
    assert artifact(configured["id"], "config") in screened["inputs"]
    assert screened["cmd"][0] == {"Literal": "srun"}
    assert any(path.name == "screen.py" for path in command_paths(screened))
    assert all(path.name != "run_screening.sh" for path in command_paths(screened))
    assert config["limit_walltime"] == 27_000
    assert screened["resources"]["walltime_seconds"] == 28_800
    assert (
        screened["resources"]["nodes"] * screened["resources"]["gpus_per_node"] == gpus
    )
    assert {output["id"]["key"]["Named"] for output in screened["outputs"]} == {
        "config",
        "trace",
        *(f"rank_{rank}" for rank in range(gpus)),
    }
