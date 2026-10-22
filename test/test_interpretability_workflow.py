"""Local table/tensor checks for paper embedding workflow adapters."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from opt.interp_embeddings.prepare import (
    analyze,
    read_cohort,
    save_embeddings,
    token_weights,
)
from opt.interp_embeddings.shallow_models import (
    linear_separability_diagnostics,
    prepare_aromaticity_paper_example,
)
from opt.interp_embeddings.prepare_cohorts import (
    canonical_and_aromatic,
    prepare_cohorts,
)


@pytest.mark.parametrize("tokenizer_source", ["model", "base", "both"])
def test_token_extraction_reads_sharded_weights_without_loading_model(
    tmp_path, tokenizer_source
):
    from safetensors.numpy import load_file, save_file

    model = tmp_path / "model"
    model.mkdir()
    key = "encoder.embeddings.word_embeddings.weight"
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    save_file({key: values}, model / "part-2.safetensors")
    # An unrelated shard is deliberately absent: extraction must not need it.
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.other.weight": "absent-part-1.safetensors",
                    key: "part-2.safetensors",
                }
            }
        )
    )
    tokenizer = {
        "model": {"vocab": {"C": 0, "O": 1, "N": 2, "S": 3, "P": 4}},
        "added_tokens": [],
    }
    fallback = tmp_path / "base-tokenizer.json"
    if tokenizer_source != "base":
        (model / "tokenizer.json").write_text(json.dumps(tokenizer))
    if tokenizer_source != "model":
        fallback.write_text(
            json.dumps(tokenizer if tokenizer_source == "base" else {"unused": True})
        )
    token_weights(
        model,
        tmp_path / "tokens",
        fallback_tokenizer=fallback if tokenizer_source != "model" else None,
    )
    extracted = load_file(tmp_path / "tokens" / "model.safetensors")
    assert list(extracted) == [key]
    np.testing.assert_array_equal(extracted[key], values)
    assert json.loads((tmp_path / "tokens" / "tokenizer.json").read_text()) == tokenizer


def test_original_shallow_helper_uses_actual_embedding_width(tmp_path):
    frame = pd.DataFrame(
        {"0": [1, 2], "1": [3, 4], "2": [5, 6], "cyclobutadiene": [0, 1]}
    )
    frame.to_csv(tmp_path / "embeddings.csv", index=False)
    embeddings, labels = prepare_aromaticity_paper_example(tmp_path / "embeddings.csv")
    assert embeddings.shape == (2, 3)
    assert labels.tolist() == [False, True]


def test_linear_diagnostics_reuse_preserves_legacy_values(monkeypatch):
    from sklearn.svm import SVC

    embeddings = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    labels = np.array([-1, -1, 1, 1])
    expected = linear_separability_diagnostics(embeddings, labels, margin=100)
    classifier = SVC(kernel="linear", C=100).fit(embeddings, labels)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Diagnostics must reuse the fitted classifier")

    monkeypatch.setattr(SVC, "fit", unexpected_fit)
    assert (
        linear_separability_diagnostics(embeddings, labels, classifier=classifier)
        == expected
    )


def test_pah_analysis_uses_supplied_validation_membership(tmp_path, monkeypatch):
    import umap
    from sklearn.svm import SVC

    fits = []
    original_fit = SVC.fit

    def counted_fit(classifier, *args, **kwargs):
        fits.append(classifier)
        return original_fit(classifier, *args, **kwargs)

    monkeypatch.setattr(SVC, "fit", counted_fit)

    class Projection:
        def __init__(self, **settings):
            self.settings = settings

        def fit_transform(self, embeddings):
            return embeddings[:, :2]

    monkeypatch.setattr(umap, "UMAP", Projection)
    frame = pd.DataFrame(
        {
            "molecule_id": list("abcdef"),
            "smiles": ["CC"] * 6,
            "split": ["train"] * 4 + ["validation"] * 2,
            "Aromaticity": ["aromatic", "other"] * 3,
            "Number of Benzene": [1, 0] * 3,
        }
    )
    frame.to_csv(tmp_path / "cohort.csv", index=False)
    embeddings = np.array(
        [
            [1, 2, 3],
            [-1, -2, -3],
            [2, 3, 4],
            [-2, -3, -4],
            [1.5, 2.5, 3.5],
            [-1.5, -2.5, -3.5],
        ]
    )
    save_embeddings(tmp_path / "embeddings.npz", embeddings, frame.molecule_id)
    with np.load(tmp_path / "embeddings.npz", allow_pickle=False) as stored:
        assert stored["molecule_id"].dtype.kind == "U"
        assert stored["molecule_id"].tolist() == list("abcdef")
    settings = {
        "umap": {
            "n_neighbors": 3,
            "min_dist": 0.1,
            "metric": "euclidean",
            "random_state": 7,
            "transform_seed": 7,
        },
        "svm": {"C": 100, "positive_label": "aromatic"},
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    args = SimpleNamespace(
        cohort=tmp_path / "cohort.csv",
        embeddings=tmp_path / "embeddings.npz",
        settings=tmp_path / "settings.json",
        panel="aromaticity",
        output_dir=tmp_path / "out",
    )
    analyze(args)
    assert len(fits) == 1
    metrics = json.loads((args.output_dir / "svm-metrics.json").read_text())
    assert metrics["validation_auroc"] == 1
    assert pd.read_csv(args.output_dir / "svm-validation.csv").molecule_id.tolist() == [
        "e",
        "f",
    ]
    assert pd.read_csv(
        args.output_dir / "interp_benzene.csv"
    ).molecule_id.tolist() == list("abcdef")
    frame.loc[5, "molecule_id"] = "a"
    frame.to_csv(args.cohort, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_cohort(args.cohort, "aromaticity")


def test_pah_aromaticity_requires_all_bonds_not_just_aromatic_atoms():
    assert canonical_and_aromatic("C1=CC=CC=C1")[1]
    assert not canonical_and_aromatic("c1ccccc1-c1ccccc1")[1]
    assert not canonical_and_aromatic("C1=CC=C1")[1]


def test_public_pah_cohorts_deduplicate_and_share_reproducible_splits(tmp_path):
    cata = [
        "c1ccc2ccccc2c1",
        "c1ccc2c(c1)ccc1ccccc12",
        "c1ccc2cc3ccccc3cc2c1",
        "c1ccc2c(c1)c1ccccc1c1ccccc21",
        "c1ccc2c(c1)ccc1c3ccccc3ccc21",
    ]
    peri = [
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        "c1ccc2c(c1)c1cccc3ccc4cccc2c4c31",
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
        "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54",
        "c1cc2ccc3cc4cccc5ccc6cc(c1)c2c3c6c54",
    ]
    other = ["C1=CC=C1", "C1CCCCC1", "C1CCCC1", "C1CCCCCCC1", "c1ccccc1-c1ccccc1"]
    paths = [tmp_path / "c1.csv", tmp_path / "c2.parquet", tmp_path / "c3.csv"]
    for path, smiles in [(paths[0], cata), (paths[2], peri + [cata[0]])]:
        pd.DataFrame(
            {
                "molecule": [str(i) for i in range(len(smiles))],
                "smiles": smiles,
                "n_rings": 4,
            }
        ).to_csv(path, index=False)
    rows = [
        {"name": str(i), "smiles": smiles, "charge": 0, "rings": 4, "benzene": 2}
        for i, smiles in enumerate(cata + other + [cata[0]])
    ]
    rows.append(
        {"name": "charged", "smiles": "C1CCC1", "charge": 1, "rings": 1, "benzene": 0}
    )
    pd.DataFrame(rows).to_parquet(paths[1], index=False)
    for directory in [tmp_path / "first", tmp_path / "second"]:
        prepare_cohorts(
            *paths, directory, seed=42, per_class=4, validation_fraction=0.25
        )
    for first in (tmp_path / "first").iterdir():
        assert first.read_bytes() == (tmp_path / "second" / first.name).read_bytes()
    for panel, label in [("aromaticity", "Aromaticity"), ("condensation", "Condensed")]:
        frame = read_cohort(tmp_path / "first" / (panel + ".csv"), panel)
        assert frame.smiles.is_unique
        assert frame[label].value_counts().tolist() == [4, 4]
        assert frame.split.value_counts().to_dict() == {"train": 6, "validation": 2}
        assert "charged" not in frame.source_id.astype(str).tolist()
        settings = json.loads(
            (tmp_path / "first" / (panel + "-settings.json")).read_text()
        )
        assert (
            settings["umap"]["random_state"] == settings["umap"]["transform_seed"] == 42
        )
        assert settings["svm"]["positive_label"] in frame[label].unique()
    condensation = pd.read_csv(tmp_path / "first" / "condensation.csv")
    assert canonical_and_aromatic(cata[0])[0] not in set(condensation.smiles)
    provenance = json.loads((tmp_path / "first" / "provenance.json").read_text())
    assert provenance["condensation_overlap_removed"] == 1
    assert provenance["counts"]["aromaticity"]["candidates"] == {
        "Aromatic": 5,
        "Not fully aromatic": 5,
    }


def test_both_pah_models_use_the_same_prepared_cohorts(inspect_workflow):
    graph = {task["id"]: task for task in inspect_workflow()}
    for prefix in ["embeddings-", "embeddings-mist-28m-"]:
        for panel in ["aromaticity", "condensation"]:
            cohort = {
                "producer": "embeddings-prepare-pah-cohorts",
                "key": {"Named": panel},
            }
            assert cohort in graph[prefix + "encode-" + panel]["inputs"]
            assert cohort in graph[prefix + "project-" + panel]["inputs"]
            assert {
                "producer": "embeddings-prepare-pah-cohorts",
                "key": {"Named": panel + "_settings"},
            } in graph[prefix + "project-" + panel]["inputs"]
