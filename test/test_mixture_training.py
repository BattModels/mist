"""Strict-fold fitting exports reloadable weights using training-only statistics."""

import json
from argparse import Namespace
from functools import partial

import pytest
import torch
import yaml
from datasets import Dataset
from smirk import SmirkTokenizerFast
from transformers import BertConfig, BertModel

from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsModel,
    _default_targets,
)
from electrolyte_fm.utils.lr_schedule import RelativeCosineWarmup
from opt.benchmarks.scripts.train_mixture import fit


@pytest.mark.parametrize("policy,configuration", [("strict", "k-compound-strict-0"), ("random", "random")])
def test_strict_fold_training_and_export(tmp_path, monkeypatch, policy, configuration):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    base = tmp_path / "base"
    tokenizer = SmirkTokenizerFast()
    BertModel(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32,
    )).save_pretrained(base)
    tokenizer.save_pretrained(base)
    architecture = tmp_path / "architecture"
    architecture.mkdir()
    (architecture / "config.json").write_text(json.dumps({
        "interactions": "difference", "num_control": 3,
        "temperature_dependence": "arrhenius", "relative_excess": False,
        "dropout": 0.05,
    }))
    fold = tmp_path / configuration
    splits = [("train", 0), ("validation", 100)]
    if policy == "random":
        splits.append(("test", 1000))
    for split, offset in splits:
        rows = []
        for index in range(4):
            row = {"smi1": "CC", "smi2": "O", "x1": 0.5, "x2": 0.5,
                   "temperature [kelvin]": 298.15 + index}
            for target in _default_targets():
                row[target] = float(offset + index + 1)
                row["excess " + target] = float(index + 1)
            rows.append(row)
        Dataset.from_list(rows).save_to_disk(fold / split)
    output = tmp_path / "fit"
    args = Namespace(dataset=fold, model=base, architecture=architecture,
                     output_dir=output, learning_rate=5e-5, batch_size=2,
                     max_steps=8, val_check_interval=1, num_workers=0,
                     max_time="00:00:05:00", split_policy=policy)
    fit(args)
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["default_callbacks"] is False
    assert config["model"]["init_args"]["model"]["class_path"].endswith("pretrained_mixture_model")
    assert config["trainer"]["callbacks"][0]["init_args"]["monitor"] == "val/loss"
    checkpoint = torch.load(output / "checkpoints/last.ckpt", weights_only=False)
    thawing = checkpoint["callbacks"]["ProgressiveThawing"]
    assert thawing["current_stage"] == 3
    assert len(thawing["lr_scale"]) == 4
    # Resume after all thawing stages, including optimizer and callback state.
    checkpoint["callbacks"]["Timer"]["time_elapsed"]["train"] = 36000
    if policy == "random":
        # Existing scheduler jobs saved direct-Trainer checkpoints, before the CLI.
        hparams = checkpoint["hyper_parameters"]
        for key in ("_class_path", "_instantiator", "vocab_size"):
            hparams.pop(key, None)
        hparams["optimizer"] = partial(torch.optim.AdamW, lr=args.learning_rate)
        hparams["lr_schedule"] = partial(
            RelativeCosineWarmup, num_training_steps=args.max_steps, num_warmup_steps="beta2"
        )
    torch.save(checkpoint, output / "checkpoints/last.ckpt")
    args.max_steps = 10
    fit(args)
    resumed = torch.load(output / "checkpoints/last.ckpt", weights_only=False)
    embeddings = "model.encoder.embeddings.word_embeddings.weight"
    assert not torch.equal(checkpoint["state_dict"][embeddings], resumed["state_dict"][embeddings])
    exported = ExcessPhysicsModel.from_pretrained(output / "model")
    assert torch.allclose(exported.transform.mean, torch.full((3,), 2.5))
    assert torch.allclose(exported.excess_transform.mean, torch.zeros(3))
    evidence = json.loads((output / "training.json").read_text())
    assert evidence["completed_steps"] == 10
    assert evidence["selection_split"] == "validation"
    assert evidence["test_data_used"] is False
    assert torch.isfinite(torch.tensor(evidence["validation_loss"]))


@pytest.mark.parametrize("policy,configuration,splits", [
    ("strict", "random", ("train", "validation")),
    ("strict", "k-compound-strict-0", ("train", "validation", "test")),
    ("random", "random", ("train", "validation")),
])
def test_split_policy_is_checked_before_training(tmp_path, policy, configuration, splits):
    dataset = tmp_path / configuration
    for split in splits:
        directory = dataset / split
        directory.mkdir(parents=True)
        (directory / "data.arrow").touch()
    args = Namespace(dataset=dataset, output_dir=tmp_path / "output", split_policy=policy)
    with pytest.raises(ValueError, match="Expected"):
        fit(args)
