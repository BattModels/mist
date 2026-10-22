"""Configure the existing probe trainer from model directories."""

import argparse
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from transformers import AutoConfig  # noqa: E402

from electrolyte_fm.models.linear_probe import cli_main  # noqa: E402


def probe_config(
    base_model,
    model,
    data,
    output_dir,
    batch_size=None,
    max_steps=10_000,
    max_epochs=-1,
    max_time="00:07:30:00",
    num_workers=4,
):
    """Configure the paper's probe fit using the acquired encoder's dimensions."""
    snapshot = str(base_model)
    if model:
        tokenizer = str(model)
        encoder = {
            "class_path": "electrolyte_fm.models.linear_probe.encoder_from_finetuned",
            "init_args": {"name_or_path": str(model), "tokenizer": tokenizer},
        }
    else:
        tokenizer = snapshot
        encoder = {
            "class_path": "electrolyte_fm.models.lm_finetuning.load_encoder",
            "init_args": {"encoder": snapshot, "tokenizer": tokenizer},
        }
    if batch_size is None:
        base_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        batch_size = 8 if base_config.hidden_size > 1024 else 128
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = str(Path(output_dir).resolve())
    return {
        "model": {
            "model": encoder,
            "probes": None,  # Infer all four hook shapes from the encoder.
            "optimizer": {
                "class_path": "torch.optim.AdamW",
                "init_args": {"lr": 0.001},
            },
            "lr_schedule": None,
        },
        "data": {
            "class_path": "electrolyte_fm.data_modules.lipinski_dataset.LipinskiDataModule",
            "init_args": {
                "path": str(data),
                "tokenizer": tokenizer,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "encoding": "smiles-kekule",
                "randomize": True,
            },
        },
        "trainer": {
            "max_steps": max_steps,
            "max_epochs": max_epochs,
            "max_time": max_time,
            "devices": 1,
            "enable_progress_bar": False,
            "default_root_dir": output,
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {"save_dir": output, "name": "metrics", "version": 0},
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--max-time", default="00:07:30:00")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    cli_main(probe_config(**vars(args)))


if __name__ == "__main__":
    main()
