import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "opt/sterochemistry/scripts/test_tmqm_selected.py"
SPEC = importlib.util.spec_from_file_location("test_tmqm_selected", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_cell(path: Path, *, model="model", lr=1e-4, loss=0.5, steps=100000, **fields):
    path.write_text(
        json.dumps(
            {
                "model": model,
                "learning_rate": lr,
                "best_validation_loss": loss,
                "selection_split": "validation",
                "test_set_accessed": False,
                "completed_optimizer_steps": steps,
                **fields,
            }
        )
    )
    return path


def test_select_validation_cell_uses_loss_then_learning_rate(tmp_path):
    cells = [
        write_cell(tmp_path / "a.json", lr=3e-5, loss=0.2),
        write_cell(tmp_path / "b.json", lr=1e-5, loss=0.2),
        write_cell(tmp_path / "c.json", lr=1e-4, loss=0.3),
    ]

    selection = MODULE.select_validation_cell(cells)

    assert selection["selected"]["learning_rate"] == 1e-5
    assert selection["test_used_for_selection"] is False


def test_select_validation_cell_rejects_incomplete_training(tmp_path):
    cells = [
        write_cell(tmp_path / "a.json", lr=1e-5),
        write_cell(tmp_path / "b.json", lr=3e-5, steps=99999),
    ]

    with pytest.raises(RuntimeError, match="ineligible validation cell"):
        MODULE.select_validation_cell(cells)


def test_selection_pairs_reports_with_declared_training_directories(tmp_path):
    runs = [tmp_path / "first", tmp_path / "second"]
    for run in runs:
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints/best.ckpt").touch()
    cells = [
        write_cell(
            tmp_path / "a.json",
            lr=1e-5,
            loss=0.5,
            best_checkpoint="checkpoints/best.ckpt",
        ),
        write_cell(
            tmp_path / "b.json",
            lr=3e-5,
            loss=0.2,
            best_checkpoint="checkpoints/best.ckpt",
        ),
    ]
    selection, checkpoint = MODULE.select_validation_run(cells, runs)
    assert selection["selected"]["learning_rate"] == 3e-5
    assert checkpoint == runs[1] / "checkpoints/best.ckpt"
    with pytest.raises(ValueError, match="one --runs directory"):
        MODULE.select_validation_run(cells, runs[:1])
    write_cell(
        cells[1],
        lr=3e-5,
        loss=0.2,
        best_checkpoint="../first/checkpoints/best.ckpt",
    )
    with pytest.raises(ValueError, match="inside its training directory"):
        MODULE.select_validation_run(cells, runs)


def test_selected_mist_model_loads_without_original_training_artifacts(
    tmp_path, monkeypatch
):
    import lightning.pytorch as pl
    import torch
    from datasets import Dataset, DatasetDict

    from electrolyte_fm.models.lm_finetuning import LMFinetuning
    from electrolyte_fm.models.prod_finetune import load_finetuned
    from test.test_feature_miner_workflow import make_snapshot

    snapshot = tmp_path / "snapshot"
    make_snapshot(snapshot)
    targets = MODULE.jsonnet_load(SCRIPT.parents[3] / "submit/tmQM.jsonnet")["train"][
        "data"
    ]["init_args"]["target_columns"]
    trained = LMFinetuning(
        output_size=len(targets),
        encoder_ckpt=str(snapshot),
        tokenizer=str(snapshot),
        task="regression",
        target_columns=targets,
        metrics=["mae"],
        track_oov=False,
    )
    trained.configure_model()
    runs, cells = [], []
    for index, lr in enumerate([1e-5, 3e-5]):
        run = tmp_path / f"run-{index}"
        (run / "checkpoints").mkdir(parents=True)
        torch.save(
            {
                "state_dict": trained.state_dict(),
                "hyper_parameters": dict(trained.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            run / "checkpoints/best.ckpt",
        )
        runs.append(run)
        cells.append(
            write_cell(
                tmp_path / f"cell-{index}.json",
                lr=lr,
                loss=0.5 - index * 0.1,
                best_checkpoint="checkpoints/best.ckpt",
                encoding="smiles",
                fit_settings={"batch_size": 2, "max_sequence_length": 64},
            )
        )
    rows = [
        {"smiles": smiles, **{target: float(index) for target in targets}}
        for index, smiles in enumerate(["CCO", "CCN", "CCCC", "CO"])
    ]
    dataset = tmp_path / "dataset"
    DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(dataset)
    output = tmp_path / "test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--cells",
            *map(str, cells),
            "--runs",
            *map(str, runs),
            "--model-dir",
            str(snapshot),
            "--label",
            "tiny",
            "--export-model",
            "--dataset",
            str(dataset),
            "--expected-test-rows",
            "4",
            "--output-dir",
            str(output),
        ],
    )
    MODULE.main()
    for run in runs:
        shutil.rmtree(run)
    shutil.rmtree(snapshot)

    loaded = load_finetuned(output / "model", local_files_only=True)
    assert loaded.encoding == "smiles"
    predicted = loaded.predict([row["smiles"] for row in rows], return_dict=False)
    saved = torch.tensor(
        [
            [json.loads(line)["prediction"][target] for target in targets]
            for line in (output / "predictions.jsonl").read_text().splitlines()
        ]
    )
    torch.testing.assert_close(predicted, saved)
