"""Offline checks for real inputs and local checkpoint loading; no Hub models."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from prepare import channels_from, performance_tables, redjade_mapping


def contract(tmp_path):
    channels = [f"descriptor_{index}" for index in range(135)]
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump({"channels": [{"name": name} for name in channels]}))
    return path, channels


def test_performance_keeps_original_membership_and_scores(tmp_path):
    schema, channels = contract(tmp_path)
    target = np.array(
        [[(row + column) % 2 for column in range(135)] for row in range(4)]
    )
    truth = pd.DataFrame(target, columns=channels)
    truth.insert(0, "smiles", ["CCO", "CC", "CCC", "CCCC"])
    truth.to_csv(tmp_path / "train.csv", index=False)
    truth.to_csv(tmp_path / "test.csv", index=False)
    predicted = truth.copy()
    predicted[channels] = target * 4.0 - 2.0
    predicted.to_csv(tmp_path / "predicted.csv", index=False)
    args = SimpleNamespace(
        contract=schema,
        train=tmp_path / "train.csv",
        test=tmp_path / "test.csv",
        predictions=tmp_path / "predicted.csv",
        output_dir=tmp_path / "out",
    )
    performance_tables(args)
    descriptor = pd.read_csv(args.output_dir / "dataset_size_auroc.csv")
    molecule = pd.read_csv(args.output_dir / "functional_group_errors.csv")
    assert (descriptor.auroc == 1).all()
    assert (descriptor.num_samples == 2).all()
    assert molecule.smiles.tolist() == truth.smiles.tolist()
    predicted.iloc[::-1].to_csv(args.predictions, index=False)
    with pytest.raises(ValueError, match="exact test molecule order"):
        performance_tables(args)


def test_redjade_missing_identity_fails_before_producing_artifact(tmp_path):
    pd.DataFrame({"RedJade Code": ["A"], "SMILES": ["CC"]}).to_csv(
        tmp_path / "public.csv", index=False
    )
    pd.DataFrame({"RedJade Code A": ["A"], "RedJade Code B": ["B"]}).to_csv(
        tmp_path / "triplets.csv", index=False
    )
    pd.DataFrame(
        columns=["redjade_code", "canonical_isomeric_smiles", "source_or_release_id"]
    ).to_csv(tmp_path / "mapping.csv", index=False)
    args = SimpleNamespace(
        public_mapping=tmp_path / "public.csv",
        mapping=tmp_path / "mapping.csv",
        triplets=tmp_path / "triplets.csv",
        output=tmp_path / "merged.csv",
    )
    with pytest.raises(ValueError, match="Missing original RedJade identities: B"):
        redjade_mapping(args)
    assert not args.output.exists()
    pd.DataFrame(
        {
            "redjade_code": ["B"],
            "canonical_isomeric_smiles": ["CCC"],
            "source_or_release_id": ["source"],
        }
    ).to_csv(args.mapping, index=False)
    redjade_mapping(args)
    assert pd.read_csv(args.output)["Sample Identifier"].tolist() == ["A", "B"]


def test_mixture_model_loads_a_local_hf_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import RobertaConfig, RobertaModel

    from electrolyte_fm.models.mixture_similarity import MixtureSimilarityModel

    encoder = RobertaModel(
        RobertaConfig(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    encoder.save_pretrained(tmp_path / "encoder")
    model = MixtureSimilarityModel(
        encoder_ckpt=str(tmp_path / "encoder"), metrics=["mae"]
    ).eval()
    assert torch.equal(
        model.encoder.embeddings.word_embeddings.weight,
        encoder.embeddings.word_embeddings.weight,
    )
    # Verify the original mixture head can use the loaded encoder.
    batch = {}
    for side in ("mix1", "mix2"):
        batch[f"input_ids_{side}"] = torch.tensor([[[0, 4, 2]]])
        batch[f"attention_mask_{side}"] = torch.ones((1, 1, 3), dtype=torch.long)
        batch[f"component_mask_{side}"] = torch.zeros((1, 1), dtype=torch.bool)
    with torch.inference_mode():
        prediction, _ = model(batch, transform=False)
    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all()


def test_native_mixture_checkpoint_path_retains_original_loader(tmp_path, monkeypatch):
    from transformers import RobertaConfig, RobertaModel

    from electrolyte_fm.models.mixture_model import DeepSpeedMixin, MixtureModel

    # A Lightning run may also have config.json; it has no HF model_type.
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {"class_path": "native.Model"}})
    )
    encoder = RobertaModel(
        RobertaConfig(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
        )
    )
    seen = []

    def load(path):
        seen.append(path)
        return SimpleNamespace(get_encoder=lambda: encoder)

    monkeypatch.setattr(DeepSpeedMixin, "load", load)
    model = MixtureModel(output_size=1, encoder_ckpt=str(tmp_path), metrics=["mae"])
    assert seen == [str(tmp_path)]
    assert model.encoder is encoder


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
def test_figures_consume_generated_splits_and_fitted_models(inspect_workflow, base):
    graph = {task["id"]: task for task in inspect_workflow("opt/olfactory/workflow.pomset")}
    prefix = "olfaction-" if base == "mist-28m" else f"olfaction-{base}-"
    mixture = graph[prefix + "mixture-predictions"]
    producers = {item["producer"] for item in mixture["inputs"]}
    assert f"benchmark-{base}-olfactory-mixtures-fit" in producers
    assert "benchmark-olfactory-mixtures-materialize-split" in producers
    held_out = graph[prefix + "held-out-predictions"]
    assert all("data/user/olfaction" not in str(value) for value in held_out["inputs"])
