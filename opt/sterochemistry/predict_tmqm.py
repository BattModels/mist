import torch
import typer
from pathlib import Path

from electrolyte_fm.models.prod_finetune import load_finetuned
from electrolyte_fm.utils.device import default_device
from datasets import load_dataset

cli = typer.Typer()


def predict_batch(x, model=None):
    with torch.inference_mode():
        y = model.predict(x)

    return {k + "_mist": v["value"] for k, v in y.items()}


@cli.command()
def predict(
    save_directory: str,
    tmqm_path: str,
    output: str = "tmqm_results.jsonl",
    device=default_device(),
    batch_size: int = 32,
    split: str = "validation",
):
    if not (Path(save_directory) / "config.json").is_file():
        raise FileNotFoundError(f"Supply a staged local tmQM model directory: {save_directory}")
    if not list(Path(tmqm_path, split).glob("*.arrow")):
        raise FileNotFoundError(f"Missing released tmQM {split} Arrow files in {tmqm_path}")
    model = load_finetuned(save_directory, local_files_only=True).eval().to(device)
    ds = load_dataset(
        "arrow",
        data_files=[str(Path(tmqm_path, split, "*.arrow"))],
        streaming=True,
    )["train"]
    ds = ds.remove_columns(["q", "MND", "xyz", "S"])

    ds = ds.map(
        predict_batch,
        input_columns="smiles",
        fn_kwargs={"model": model},
        batched=True,
        batch_size=batch_size,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(output)


if __name__ == "__main__":
    cli()
