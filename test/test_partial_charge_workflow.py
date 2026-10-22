"""Exercise charge fitting and prediction with small, entirely local models/data."""

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from datasets import Dataset, DatasetDict
from rdkit import Chem
from smirk import SmirkTokenizerFast
from transformers import RobertaConfig, RobertaModel

from electrolyte_fm.data_modules.pubchem_qc import (
    DEFAULT_SEQ_TARGETS,
    DEFAULT_TOKEN_TARGETS,
    PubChemQC,
    mol_from_prediction,
)
from opt.benchmarks.scripts import partial_charge as workflow
from opt.benchmarks.scripts.materialize_pubchemqc import materialize


def test_charge_cli_resolves_workflow_namespace_without_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Exercise the workflow launcher with this test's already prepared venv.
    environment["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    result = subprocess.run(
        ["uv", "run", "--no-sync", "--offline", "--project", str(root / "opt/benchmarks"),
         "--python", sys.executable, "--directory", str(root),
         "-m", "opt.benchmarks.scripts.partial_charge", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fit" in result.stdout and "evaluate" in result.stdout


@pytest.fixture
def local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    path = tmp_path / "base"
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    tokenizer.save_pretrained(path)
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    RobertaModel(config, add_pooling_layer=False).save_pretrained(path)
    return path, tokenizer


def molecule_record(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    n = mol.GetNumAtoms()
    return {
        "atomic-numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
        "atomic-coordinates": [0.0] * (3 * n),
        "bond-connections": [
            i for b in mol.GetBonds() for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        ],
        "bond-order": [int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()],
        **{
            target: [0.1 * (i - n / 2) for i in range(n)]
            for target in DEFAULT_TOKEN_TARGETS
        },
        **{target: 0.0 for target in DEFAULT_SEQ_TARGETS},
    }


@pytest.fixture
def local_data(tmp_path):
    rows = [molecule_record(smi) for smi in ("O", "CO", "CCO", "CCN")]
    dataset = tmp_path / "data"
    DatasetDict(
        {name: Dataset.from_list(rows) for name in ("train", "validation", "test")}
    ).save_to_disk(dataset)
    artifact = tmp_path / "materialized/dataset"
    materialize(dataset, artifact, artifact.parent / "receipt.json")
    return dataset, artifact


def test_charge_corpus_artifact_loads_after_relocation_without_original_source(
    tmp_path, local_data, local_snapshot
):
    source, artifact = local_data
    snapshot, _ = local_snapshot
    assert artifact.is_symlink()
    assert not os.path.isabs(os.readlink(artifact))
    original = next((source / "train").glob("*.arrow"))
    assert (artifact / "train" / original.name).stat().st_ino == original.stat().st_ino
    inode = artifact.lstat().st_ino
    materialize(source, artifact)
    assert artifact.lstat().st_ino == inode
    exported = tmp_path / "relocated"
    shutil.copytree(artifact.parent, exported)
    shutil.rmtree(source)
    shutil.rmtree(artifact.parent)
    metadata = json.loads((exported / "receipt.json").read_text())
    assert metadata["dataset"] == "dataset"
    assert all(
        (exported / "dataset" / name).is_file()
        for paths in metadata["splits"].values()
        for name in paths
    )
    data = workflow.data_module(exported / "dataset", str(snapshot), 2, workers=0)
    data.setup("fit")
    assert len(list(data.val_dataset)) == len(list(data.test_dataset)) == 4
    batch = next(iter(data.test_dataloader()))
    assert batch["token_target_mask"].any()


def test_charge_annotations_without_geometry(local_snapshot):
    _, tokenizer = local_snapshot
    ids = tokenizer("CO", return_tensors="pt")["input_ids"][0]
    values = torch.tensor([-0.2, 0.1, -0.3, 0.15]).repeat(len(ids), 1)
    mol = mol_from_prediction(
        ids, tokenizer, y_token=values, token_targets=DEFAULT_TOKEN_TARGETS
    )
    assert mol.GetNumConformers() == 0
    for atom in mol.GetAtoms():
        assert atom.GetDoubleProp("partial-charge-mulliken") == pytest.approx(-0.2)
        assert atom.GetDoubleProp("hs_partial-charge-lowdin") == pytest.approx(0.15)


def test_charge_predictions_preserve_stereochemical_identity(local_snapshot):
    snapshot, tokenizer = local_snapshot
    smiles = [
        "O=C1O[C@@H](F)[C@@H](F)O1",
        "O=C1OC(F)C(F)O1",
        "O=C1O[C@H](F)[C@@H](F)O1",
        "O=C1O[C@@H](F)[C@H](F)O1",
        "F/C=C/F",
        "F/C=C\\F",
    ]
    expected = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]
    model = workflow.build_model(str(snapshot)).eval()
    rows = list(workflow.predict_rows(model, tokenizer, smiles, 2))
    assert [row["smi"] for row in rows] == expected
    assert len({row["smi"] for row in rows}) == len(smiles)


def test_charge_encoder_preserves_position_offset(local_snapshot):
    path, tokenizer = local_snapshot
    model = workflow.build_model(str(path))
    batch = tokenizer(["CO", "O"], padding=True, return_tensors="pt")
    positions = []
    handle = model.encoder.embeddings.position_embeddings.register_forward_pre_hook(
        lambda module, args: positions.append(args[0].detach().clone())
    )
    model(**batch)
    handle.remove()
    mask = batch["input_ids"].ne(tokenizer.pad_token_id).long()
    assert torch.equal(positions[0], mask.cumsum(-1) * mask + 1)
    assert model.encoder.config.pad_token_id == tokenizer.pad_token_id
    assert model.sequence_loss_weight == 0


def test_charge_beta_recipe_warms_up_from_optimizer_beta2(local_snapshot):
    snapshot, _ = local_snapshot
    original = workflow.build_model(str(snapshot)).configure_optimizers()
    assert isinstance(original, torch.optim.AdamW)
    assert original.defaults["betas"] == (0.9, 0.999)
    updated = workflow.build_model(
        str(snapshot), learning_rate=1e-4, betas=(0.87, 0.997),
        warmup_steps="beta2", max_steps=100000,
    ).configure_optimizers()
    optimizer, scheduler = updated["optimizer"], updated["lr_scheduler"]["scheduler"]
    assert optimizer.defaults["betas"] == (0.87, 0.997)
    assert updated["lr_scheduler"]["interval"] == "step"
    assert scheduler.get_last_lr() == [0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1e-4 / 667])
    for _ in range(666):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1e-4])


def test_validation_encoding_is_deterministic(local_data, local_snapshot):
    path, _ = local_data
    snapshot, _ = local_snapshot
    data = PubChemQC(
        str(path), tokenizer=str(snapshot), num_workers=0, evaluation_randomize=False
    )
    data.setup("fit")
    first = [row["input_ids"] for row in data.val_dataset]
    second = [row["input_ids"] for row in data.val_dataset]
    assert len(first) == len(second) == 4
    assert all(
        torch.equal(torch.as_tensor(a), torch.as_tensor(b))
        for a, b in zip(first, second, strict=True)
    )


def test_fit_select_restore_and_predict(tmp_path, local_snapshot, local_data):
    snapshot, tokenizer = local_snapshot
    _, dataset = local_data
    args = Namespace(
        model=snapshot,
        dataset=dataset,
        learning_rate=1e-3,
        task="pubchemqc-tiny",
        batch_size=2,
        num_workers=0,
        output_dir=tmp_path / "fit",
        gradient_accumulation=1,
        max_steps=4,
        max_time="00:00:02:00",
        val_check_interval=2,
    )
    initial = (
        workflow.build_model(str(snapshot))
        .encoder.get_input_embeddings()
        .weight.detach()
        .clone()
    )
    workflow.fit(args)
    cell = args.output_dir / "cell.json"
    selected, selected_model = workflow.select_cell([cell], [args.output_dir / "model"])
    assert selected_model == args.output_dir / "model"
    assert selected["test_data_used"] is False
    assert selected["validation_loss"] >= 0
    assert sorted(p.name for p in (args.output_dir / "checkpoints").glob("*.ckpt")) == [
        "best.ckpt",
        "last.ckpt",
    ]
    selection = tmp_path / "selection.json"
    workflow.write_json(selection, {"selected": selected})
    model, loaded_tokenizer = workflow.restore(args.output_dir / "model")
    assert not torch.equal(initial, model.encoder.get_input_embeddings().weight)
    assert model.token_targets == selected["targets"]
    predictions = list(workflow.predict_rows(model, loaded_tokenizer, ["CO", "O"], 2))
    assert len(predictions) == 2
    assert all(
        "partial-charge-lowdin" in atom["properties"]
        for row in predictions
        for atom in row["atoms"]
    )
    reloaded, _ = workflow.restore(args.output_dir / "model")
    inputs = tokenizer(["CO", "O"], padding=True, return_tensors="pt")
    with torch.no_grad():
        assert torch.equal(model(**inputs)[1], reloaded(**inputs)[1])
    workflow.evaluate(
        Namespace(
            cells=[cell],
            models=[args.output_dir / "model"],
            output_dir=tmp_path / "test",
            task=args.task,
            base="tiny",
            dataset=dataset,
            batch_size=2,
            num_workers=0,
        )
    )
    result = json.loads((tmp_path / "test" / "metrics.json").read_text())
    assert result["rows"] == 4
    assert result["split"] == "test"
    assert result["metrics"]["mae"] >= 0
    portable = tmp_path / "portable"
    shutil.copytree(tmp_path / "test/model", portable)
    args.max_steps = 6
    workflow.fit(args)
    resumed = json.loads(cell.read_text())
    assert resumed["global_step"] == 6
    assert len(list((args.output_dir / "checkpoints").glob("*.ckpt"))) == 2
    shutil.rmtree(snapshot)
    shutil.rmtree(args.output_dir)
    (
        relocated,
        _,
    ) = workflow.restore(portable)
    with torch.no_grad():
        assert torch.equal(model(**inputs)[1], relocated(**inputs)[1])


def test_selection_rejects_test_tuned_cell(tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.touch()
    cell = tmp_path / "cell.json"
    workflow.write_json(
        cell,
        {
            "checkpoint": str(checkpoint),
            "validation_loss": 0.1,
            "selection_split": "test",
            "test_data_used": True,
        },
    )
    with pytest.raises(ValueError, match="validation only"):
        workflow.select_cell([cell], [tmp_path / "model"])


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([(0.2, 1e-4), (0.1, 2e-4)], 1),
        ([(0.1, 2e-4), (0.1, 1e-4)], 1),
        ([(0.1, 1e-4), (0.1, 1e-4)], 0),
    ],
)
def test_selection_keeps_model_association_and_tie_breaks(tmp_path, scores, expected):
    cells = [tmp_path / f"cell-{index}.json" for index in range(len(scores))]
    models = [tmp_path / f"model-{index}" for index in range(len(scores))]
    for path, (loss, learning_rate) in zip(cells, scores, strict=True):
        workflow.write_json(path, {
            "task": "pubchemqc-tiny", "selection_split": "validation",
            "test_data_used": False, "validation_loss": loss,
            "learning_rate": learning_rate,
        })
    selected, model = workflow.select_cell(cells, models)
    assert model == models[expected]
    assert (selected["validation_loss"], selected["learning_rate"]) == scores[expected]


@pytest.mark.parametrize("transform", ["identity", "standardize"])
def test_production_model_reloads_finetuning_checkpoint(
    tmp_path, local_snapshot, transform
):
    from electrolyte_fm.models.lm_finetuning import LMFinetuning
    from electrolyte_fm.models.prod_finetune import load_finetuned
    from opt.benchmarks.scripts.checkpoint_selection import export_model

    snapshot, tokenizer = local_snapshot
    model = LMFinetuning(
        output_size=2,
        encoder_ckpt=str(snapshot),
        tokenizer=str(snapshot),
        task="regression",
        target_columns=["homo", "gap"],
        metrics=["mae"],
        transform=transform,
        track_oov=False,
    )
    model.configure_model()
    if transform == "standardize":
        model.transform.mean.copy_(torch.tensor([-0.2, 0.3]))
        model.transform.std.copy_(torch.tensor([0.05, 0.1]))
    model.eval()
    checkpoint = tmp_path / "selected.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    packaged = tmp_path / "packaged"
    export_model(checkpoint, snapshot, packaged)
    selected = load_finetuned(packaged, local_files_only=True)
    smiles = ["CCO", "OCC", "O"]
    with torch.no_grad():
        expected = model(tokenizer(smiles, padding=True, return_tensors="pt"))
    actual = selected.predict(smiles)
    assert selected.channels == ["homo", "gap"]
    assert torch.allclose(torch.tensor(actual["homo"]["value"]), expected[:, 0])
    assert all(values["value"].numel() == 0 for values in selected.predict([]).values())


def test_probes_keep_encoder_frozen_and_use_correct_positions(local_snapshot):
    from electrolyte_fm.models.linear_probe import LightningProbe, probe_everything

    snapshot, tokenizer = local_snapshot
    encoder = workflow.build_model(str(snapshot)).encoder
    probes = LightningProbe(encoder, probe_everything(8, 5, 1, 16))
    assert len(probes.hookpoints) == 4
    positions = []
    handle = encoder.embeddings.position_embeddings.register_forward_pre_hook(
        lambda module, args: positions.append(args[0].detach().clone())
    )
    batch = tokenizer(["CCO", "O"], padding=True, return_tensors="pt")
    original = encoder.get_input_embeddings().weight.detach().clone()
    loss = probes.forward_fit(
        {**batch, "target": torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])}
    )["loss"]
    loss.backward()
    torch.optim.AdamW(probes._probes.parameters()).step()
    assert torch.equal(original, encoder.get_input_embeddings().weight)
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is not None for p in probes._probes.parameters())
    mask = batch["input_ids"].ne(tokenizer.pad_token_id).long()
    assert torch.equal(positions[0], mask.cumsum(-1) * mask + 1)
    handle.remove()
