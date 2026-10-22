"""Offline checks for resolved pairs and a tiny selected, reloadable mixture fit."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from .materialize_olfactory_mixtures import materialize, observations


def source_tables():
    definitions = pd.DataFrame({
        "Dataset": ["A", "A", "B"], "Mixture Label": [1, 2, 1],
        "CID_1": [10, 20, 20], "CID_2": [0, np.nan, 10],
    })
    annotations = pd.DataFrame({
        "Dataset": ["A"] * 19 + ["B"], "Mixture 1": [1] * 20,
        "Mixture 2": [2] * 19 + [1], "Experimental Values": np.linspace(0.1, 0.9, 20),
    })
    identities = pd.DataFrame({"CID": [10, 20], "Canonical SMILES": ["CC", "CCO"]})
    return definitions, annotations, identities


def test_padding_is_ignored_but_unknown_cids_fail():
    definitions, annotations, identities = source_tables()
    frame = observations(definitions, annotations, identities)
    assert len(frame) == len(annotations)
    assert json.loads(frame.iloc[0].mix1_smiles) == ["CC"]
    assert json.loads(frame.iloc[-1].mix1_smiles) == ["CCO", "CC"]
    definitions.loc[0, "CID_2"] = 999
    with pytest.raises(ValueError, match="Missing molecular identities"):
        observations(definitions, annotations, identities)


def test_split_preserves_all_observations_deterministically(tmp_path):
    for name, frame in zip(("definitions", "annotations", "identities"), source_tables(), strict=True):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    args = SimpleNamespace(
        **{name: tmp_path / f"{name}.csv" for name in ("definitions", "annotations", "identities")},
        output_dir=tmp_path / "split", seed=42, expected_rows=20,
    )
    materialize(args)
    original = {path.name: path.read_bytes() for path in args.output_dir.iterdir()}
    materialize(args)
    assert original == {path.name: path.read_bytes() for path in args.output_dir.iterdir()}
    splits = [pd.read_csv(args.output_dir / f"{name}.csv") for name in ("train", "val", "test")]
    assert [len(frame) for frame in splits] == [16, 2, 2]
    assert sorted(pd.concat(splits).observation_id) == list(range(20))
    metadata = json.loads((args.output_dir / "split.json").read_text())
    assert metadata["entity_disjoint"] is False


def test_tiny_fit_publishes_selected_checkpoint_with_raw_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    import torch
    from .train_olfactory_mixtures import fit
    from transformers import RobertaConfig, RobertaModel

    from electrolyte_fm.data_modules import MultiMixtureDataModule
    from electrolyte_fm.models import MixtureSimilarityModel
    from electrolyte_fm.utils.tokenizer import load_tokenizer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = tmp_path / "encoder"
    tokenizer = load_tokenizer("character")
    tokenizer.save_pretrained(base)
    RobertaModel(RobertaConfig(
        vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=256,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        pad_token_id=tokenizer.pad_token_id,
    )).save_pretrained(base)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    frame = pd.DataFrame({
        "mix1_smiles": ['["CC"]', '["CCO"]', '["CCC"]', '["CC", "CO"]'],
        "mix2_smiles": ['["CO"]', '["CC"]', '["CCO"]', '["CCC"]'],
        "value": [0.1, 0.8, 0.2, 0.7],
    })
    for name in ("train", "val", "test"):
        frame.to_csv(dataset / f"{name}.csv", index=False)
    args = SimpleNamespace(
        dataset=dataset, model=base, output_dir=tmp_path / "fit", batch_size=2,
        val_batch_size=2, num_workers=0, learning_rate=1.6e-4,
        max_steps=2, val_check_interval=1, precision="32-true",
        optimizer_offload=False, max_time="00:00:05:00",
    )
    fit(args)
    config = yaml.safe_load((args.output_dir / "config.yaml").read_text())
    assert config["seed_everything"] == 42
    assert config["default_callbacks"] is False
    assert config["model"]["class_path"] == "electrolyte_fm.models.MixtureSimilarityModel"
    assert config["model"]["init_args"]["transform"] == "identity"
    assert config["trainer"]["accelerator"] == "cpu"
    # A second invocation must resume the saved CLI checkpoint and advance it.
    args.max_steps = 3
    fit(args)
    latest = torch.load(args.output_dir / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    assert latest["global_step"] == 3
    assert latest["optimizer_states"]
    checkpoint = args.output_dir / "model/model.ckpt"
    assert checkpoint.is_file()
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "optimizer_states" not in selected
    training = json.loads((args.output_dir / "training.json").read_text())
    assert training["test_evaluated"] is False
    assert training["completed_steps"] == 3
    model = MixtureSimilarityModel.load_from_checkpoint(
        checkpoint, encoder_ckpt=str(base), weights_only=False,
    ).eval()
    data = MultiMixtureDataModule(
        path=dataset, tokenizer=str(base), target_columns=["value"],
        batch_size=2, val_batch_size=2, encoding="smiles-kekule", num_workers=0,
    )
    data.setup("validate")
    batch = next(iter(data.val_dataloader()))
    assert data.val_dataset[0]["compounds_mix1"] == ["CC"]
    padded = data.collate_fn([data.val_dataset[0], data.val_dataset[3]], randomize=False)
    assert padded["component_mask_mix1"].tolist() == [[False, True], [False, False]]
    assert torch.allclose(model.transform.inverse(batch["target"]), batch["target"])
    with torch.inference_mode():
        prediction, _ = model(batch, transform=False)
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
