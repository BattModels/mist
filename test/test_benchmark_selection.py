import json
from pathlib import Path

import pytest
import torch

from opt.benchmarks.scripts.checkpoint_selection import best_checkpoint
from opt.benchmarks.scripts.select_model import select


def save_checkpoint(path: Path, score: float) -> None:
    torch.save(
        {
            "callbacks": {
                "checkpoint": {
                    "monitor": "val/loss_epoch",
                    "best_k_models": {
                        str(path): torch.tensor(score, dtype=torch.float64)
                    },
                }
            }
        },
        path,
    )


def test_selects_saved_score_instead_of_newest_or_rounded_name(tmp_path):
    best = tmp_path / "epoch=0-step=1-val_loss=0.12.ckpt"
    newer = tmp_path / "epoch=0-step=2-val_loss=0.12.ckpt"
    save_checkpoint(best, 0.121)
    save_checkpoint(newer, 0.124)
    selected = best_checkpoint(tmp_path)
    assert selected["checkpoint"] == str(best)
    assert selected["validation_loss"] == pytest.approx(0.121)


def test_does_not_select_missing_or_nonfinite_checkpoint(tmp_path):
    path = tmp_path / "invalid.ckpt"
    save_checkpoint(path, float("nan"))
    with pytest.raises(RuntimeError, match="finite validation"):
        best_checkpoint(tmp_path)


def test_validation_selection_exposes_winning_model_without_test_input(tmp_path):
    cells, models = [], []
    for index, loss in enumerate([0.2, 0.1]):
        cell = tmp_path / f"cell-{index}.json"
        cell.write_text(json.dumps({
            "validation_loss": loss, "learning_rate": 1e-4,
            "selection_split": "validation", "test_data_used": False,
        }))
        model = tmp_path / f"model-{index}"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"selected": index}))
        cells.append(cell)
        models.append(model)
    output = tmp_path / "selected"
    select(cells, models, output)
    assert json.loads((output / "model/config.json").read_text()) == {"selected": 1}
    assert (output / "model/config.json").is_symlink()
    assert json.loads((output / "selection.json").read_text())["test_data_used"] is False
