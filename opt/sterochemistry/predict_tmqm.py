import torch
import typer
from pathlib import Path

from electrolyte_fm.models.prod_finetune import MISTFinetuned
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
    model = MISTFinetuned.from_pretrained(save_directory).eval().to(device)
    ds = load_dataset(
        "arrow",
        name=tmqm_path,
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
    ds.to_json(output)


if __name__ == "__main__":
    cli()
