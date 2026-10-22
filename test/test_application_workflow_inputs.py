"""Original-cohort guards and local model inference for isotope/tmQM figure inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from opt.isotopes import prepare


def protocol_file(tmp_path, **overrides):
    protocol = dict(encoding="smiles", pooling="cls", projection_columns="embeddings",
                    n_neighbors=10, min_dist=2, metric="manhattan", seed=42,
                    tol_offset=0.8, tol_scale=0.02, umap_kwargs={})
    protocol.update(overrides)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path


def cohort_file(tmp_path):
    path = tmp_path / "cohort.csv"
    pd.DataFrame({"smiles": ["[14C]", "[3H]", "[235U]"],
                  "neutrons": [8, 2, 143], "protons": [6, 1, 92]}).to_csv(path, index=False)
    return path


def test_isotope_cohort_preserves_order_and_checks_requested_counts(tmp_path):
    path = cohort_file(tmp_path)
    assert prepare.read_cohort(path, expected_rows=3).smiles.tolist() == ["[14C]", "[3H]", "[235U]"]
    with pytest.raises(ValueError, match="925"):
        prepare.read_cohort(path, expected_rows=925)
    assert len(prepare.read_cohort(path)) == 3
    frame = pd.read_csv(path)
    frame.loc[0, "neutrons"] = 7
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="agree"):
        prepare.read_cohort(path, expected_rows=3)


@pytest.mark.parametrize("choices", [
    {"projection_columns": None}, {"pooling": "mean"}, {"metric": "unknown"},
    {"tol_scale": float("nan")}, {"umap_kwargs": {"min_dist": 1}},
])
def test_isotope_protocol_does_not_guess_ambiguous_settings(tmp_path, choices):
    with pytest.raises(ValueError):
        prepare.read_protocol(protocol_file(tmp_path, **choices))


def test_missing_isotope_inputs_name_real_required_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing isotope cohort"):
        prepare.read_cohort(tmp_path / "cohort.csv")
    with pytest.raises(FileNotFoundError, match="Missing isotope embedding/projection"):
        prepare.read_protocol(tmp_path / "protocol.json")


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    import torch
    from smirk import SmirkTokenizerFast
    from transformers import BertConfig, BertModel

    from electrolyte_fm.models.normalize import IdentityTransform
    from electrolyte_fm.models.prediction_task_head import PredictionTaskHead
    from electrolyte_fm.models.prod_finetune import MISTFinetuned

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(16)
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  max_position_embeddings=32, pad_token_id=tokenizer.pad_token_id),
                        add_pooling_layer=False)
    model = MISTFinetuned.from_components(encoder, PredictionTaskHead(8, 2, dropout=0),
                                        IdentityTransform(2), tokenizer=tokenizer,
                                        channels=["Metal_q", "HL_Gap"]).eval()
    path = tmp_path / "model"
    model.save_pretrained(path)
    return path, model


def test_isotope_embeddings_round_trip_through_local_task_model(tmp_path, monkeypatch, local_model):
    path, model = local_model
    cohort = cohort_file(tmp_path)
    original_reader = prepare.read_cohort
    monkeypatch.setattr(prepare, "read_cohort", lambda value: original_reader(value, expected_rows=3))
    output = tmp_path / "outputs/embeddings.csv"
    prepare.encode(SimpleNamespace(cohort=cohort, protocol=protocol_file(tmp_path), model=path,
                                   output=output, device="cpu", batch_size=2))
    frame = pd.read_csv(output)
    assert frame.smiles.tolist() == ["[14C]", "[3H]", "[235U]"]
    assert frame.columns.tolist() == ["smiles", "neutrons", "protons", *[f"embedding_{i}" for i in range(8)]]
    expected = model.embed(frame.smiles.tolist(), batch_size=1).numpy()
    np.testing.assert_allclose(frame.iloc[:, 3:].to_numpy(), expected, atol=1e-6)


def test_tmqm_prediction_keeps_identifiers_and_reference_values(tmp_path, local_model):
    from datasets import Dataset
    from opt.sterochemistry.predict_tmqm import predict

    path, model = local_model
    dataset = tmp_path / "tmqm"
    Dataset.from_dict(dict(smiles=["C", "CC"], id=["LEYLAC", "LIMTIK"],
                           Metal_q=[0.2, 0.3], HL_Gap=[0.01, 0.02],
                           q=[0, 0], MND=[1, 1], xyz=["", ""], S=[0, 0])).save_to_disk(dataset / "validation")
    output = tmp_path / "prediction-output/predictions.jsonl"
    predict(str(path), str(dataset), output=str(output), device="cpu", batch_size=1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["LEYLAC", "LIMTIK"]
    assert [row["Metal_q"] for row in rows] == [0.2, 0.3]
    expected = model.predict(["C", "CC"], batch_size=1)
    for channel in ["Metal_q", "HL_Gap"]:
        np.testing.assert_allclose([row[channel + "_mist"] for row in rows], expected[channel]["value"].numpy())
