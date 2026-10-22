"""Conductivity fitting resumes and exports the validation-selected weights."""

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from lightning.pytorch.callbacks import Callback
from smirk import SmirkTokenizerFast
from transformers import BertConfig, BertModel

from electrolyte_fm.models.prod_mixture import MISTIonicConductivity
from opt.benchmarks.scripts import train_conductivity


class StopEarly(Callback):
    """Interrupt the actual CLI and reproduce a stale periodic checkpoint."""

    def on_fit_start(self, trainer, pl_module):
        assert pl_module.encoder.is_gradient_checkpointing
        assert trainer.optimizers[0].param_groups[0]["lr"] == 3e-4

    def on_train_batch_end(self, trainer, *args):
        if trainer.global_step == 2:
            self.stale_last = torch.load(
                Path(trainer.default_root_dir) / "checkpoints/last.ckpt", weights_only=False
            )
            trainer.should_stop = True

    def on_train_end(self, trainer, *args):
        torch.save(self.stale_last, Path(trainer.default_root_dir) / "checkpoints/last.ckpt")

    def on_test_start(self, *args):
        raise AssertionError("Fitting must not evaluate the test split")


def test_load_selected_deepspeed_checkpoint(tmp_path):
    checkpoint = tmp_path / "selected.ckpt"
    shard = checkpoint / "checkpoint"
    shard.mkdir(parents=True)
    (checkpoint / "latest").write_text("checkpoint")
    weights = torch.tensor([1.25, 2.5, 3.75, 4.0])
    torch.save({"optimizer_state_dict": {
        "zero_stage": 2, "partition_count": 1,
        "single_partition_of_fp32_groups": [weights],
    }}, shard / "zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save({
        "module": {}, "buffer_names": [],
        "param_shapes": [{"encoder.weight": torch.Size([2, 2])}],
        "shared_params": {}, "global_step": 7,
    }, shard / "mp_rank_00_model_states.pt")
    selected = train_conductivity.load_selected(checkpoint)
    assert selected["global_step"] == 7
    torch.testing.assert_close(selected["state_dict"]["encoder.weight"], weights.reshape(2, 2))
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("legacy_hparams", [False, True])
def test_fresh_training_resume_and_export(tmp_path, monkeypatch, legacy_hparams):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    base = tmp_path / "base"
    tokenizer = SmirkTokenizerFast()
    BertModel(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
    )).save_pretrained(base)
    tokenizer.save_pretrained(base)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split, offset in (("train", 0), ("val", 1), ("test", 1000)):
        rows = []
        for index in range(4):
            row = {"temperature": 295.0 + index, "ln k": float(index + offset)}
            for component, smiles in enumerate(("CC", "O", "CO", "[F-]", "[Li+]"), 1):
                row[f"smi{component}"] = smiles
                row[f"x{component}"] = 0.2
            rows.append(row)
        pd.DataFrame(rows).to_csv(dataset / f"{split}.csv", index=False)
    output = tmp_path / "fit"
    args = Namespace(dataset=dataset, model=base, output_dir=output,
                     max_steps=4, val_check_interval=1, num_workers=0,
                     max_time="00:00:05:00", precision="32-true")

    config_builder = train_conductivity.training_config

    def interrupted_config(args):
        config = config_builder(args)
        config["trainer"]["callbacks"].append({"class_path": "test.test_conductivity_training.StopEarly"})
        return config

    monkeypatch.setattr(train_conductivity, "training_config", interrupted_config)
    with pytest.raises(RuntimeError, match="resume"):
        train_conductivity.fit(args)
    assert (output / "checkpoints/last.ckpt").is_file()
    assert not (output / "model").exists()
    assert not (output / "training.json").exists()
    last = output / "checkpoints/last.ckpt"
    interrupted = torch.load(last, weights_only=False)
    assert interrupted["global_step"] == 2
    if legacy_hparams:
        # Pre-CLI conductivity fits saved flat model arguments and omitted
        # optimizer factories. Their optimizer/loop state must still resume.
        assert "_class_path" in interrupted["hyper_parameters"]
        interrupted["hyper_parameters"] = {
            key: value for key, value in interrupted["hyper_parameters"].items()
            if not key.startswith("_") and key not in {"optimizer", "lr_schedule"}
        }
    interrupted["callbacks"]["Timer"]["time_elapsed"]["train"] = 36000
    torch.save(interrupted, last)
    monkeypatch.setattr(train_conductivity, "training_config", config_builder)
    train_conductivity.fit(args)

    evidence = json.loads((output / "training.json").read_text())
    assert evidence["completed_steps"] == 4
    assert evidence["selection_split"] == "validation"
    assert evidence["test_data_used"] is False
    selected = torch.load(evidence["checkpoint"], weights_only=False, map_location="cpu")
    assert selected["global_step"] == evidence["selected_step"]
    exported = MISTIonicConductivity.from_pretrained(output / "model")
    assert exported.n_components == 5
    for key, value in exported.state_dict().items():
        torch.testing.assert_close(value, selected["state_dict"][key], rtol=0, atol=0)
    assert (output / "model/tokenizer.json").is_file()
