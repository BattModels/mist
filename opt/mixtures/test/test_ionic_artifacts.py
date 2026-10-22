"""Generated ionic tables carry model dependencies through to both figures."""

import pytest


@pytest.mark.parametrize(
    "base,model,main,angell",
    [
        ("mist-28m", "ionic-stage-released-model", "ionic-paper-main", "ionic-angell"),
        (
            "mist-1p8b",
            "benchmark-mist-1p8b-ionic-conductivity-fit",
            "ionic-mist-1p8b-saved-curves",
            "ionic-mist-1p8b-angell",
        ),
    ],
)
def test_ionic_figures_consume_generated_model_artifacts(inspect_workflow, base, model, main, angell):
    graph = {task["id"]: task for task in inspect_workflow("opt/mixtures/ionic_workflow.pomset")}
    producer = f"ionic-{base}-prediction-tables"
    assert {"producer": model, "key": {"Named": "model"}} in graph[producer]["inputs"]
    assert {output["path"] for output in graph[producer]["outputs"]} == {
        f"//results/ionic-conductivity/{base}/predictions/ionic_conductivity_curves.csv",
        f"//results/ionic-conductivity/{base}/predictions/ionic_conductivity_inference.csv",
    }
    for figure, artifact in [(main, "curves"), (angell, "inference")]:
        assert {"producer": producer, "key": {"Named": artifact}} in graph[figure]["inputs"]
        assert "figures" in graph[figure]["tags"]
        assert "data/user" not in str(graph[figure]["cmd"])
