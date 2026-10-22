"""Match full-precision validation scores to the checkpoints that earned them."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def load_selected(path):
    """Load selected Lightning weights, consolidating a DeepSpeed checkpoint."""
    if Path(path).is_dir():
        from lightning.pytorch.utilities.deepspeed import (
            convert_zero_checkpoint_to_fp32_state_dict,
        )

        with TemporaryDirectory(dir=Path(path).parent) as temporary:
            return convert_zero_checkpoint_to_fp32_state_dict(path, Path(temporary) / "selected.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_metadata(path):
    """Read training and callback state without consolidating DeepSpeed weights."""
    path = Path(path)
    if path.is_dir():
        tag = (path / "latest").read_text().strip()
        path = next(iter(sorted((path / tag).glob("*model_states.pt"))))
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def select_lr(cells):
    """Return the winning index using validation loss, then LR to break ties."""
    if not cells or any(
        cell.get("test_data_used") is not False
        or cell.get("selection_split") != "validation"
        for cell in cells
    ):
        raise ValueError("LR cells do not prove validation-only selection")
    if any(not math.isfinite(cell["validation_loss"]) for cell in cells):
        raise ValueError("LR cells require finite validation losses")
    return min(range(len(cells)), key=lambda i: (cells[i]["validation_loss"], cells[i]["learning_rate"]))


def export_model(checkpoint, base_model, output, *, encoding="smiles-kekule"):
    """Save the validation-selected weights and tokenizer for standalone loading."""
    from electrolyte_fm.models.prod_finetune import load_finetuned

    model = load_finetuned(checkpoint, tokenizer=base_model, local_files_only=True)
    model.config.encoding = encoding
    model.save_pretrained(output)


def best_checkpoint(directory: Path) -> dict:
    candidates = {}
    for path in sorted({path.resolve() for path in directory.rglob("*.ckpt")}):
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for state in checkpoint.get("callbacks", {}).values():
            if state.get("monitor") != "val/loss_epoch":
                continue
            for filename, value in state.get("best_k_models", {}).items():
                selected = Path(filename).resolve()
                score = float(value)
                if (
                    selected.is_file()
                    and selected.is_relative_to(directory.resolve())
                    and math.isfinite(score)
                ):
                    candidates[selected] = score
        del checkpoint
    if not candidates:
        raise RuntimeError(
            f"No saved checkpoint with a finite validation score in {directory}"
        )
    path, score = min(candidates.items(), key=lambda item: (item[1], str(item[0])))
    return {"checkpoint": str(path), "validation_loss": score}
