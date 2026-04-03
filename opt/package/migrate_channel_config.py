#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import typer
from huggingface_hub import hf_hub_download

from .channel_schema import resolve_dataset_channels

cli = typer.Typer()


def resolve_config_source(source: str) -> Tuple[Path, str]:
    source_path = Path(source)

    if source_path.is_dir():
        config_path = source_path / "config.json"
        if not config_path.is_file():
            raise typer.BadParameter(f"No config.json found in directory: {source}")
        return config_path, str(source_path)

    if source_path.is_file():
        return source_path, str(source_path)

    try:
        config_path = hf_hub_download(repo_id=source, filename="config.json")
    except Exception as exc:
        raise typer.BadParameter(
            f"Could not resolve '{source}' as a local config or Hugging Face model ID"
        ) from exc

    return Path(config_path), source


@cli.command()
def main(
    source: str = typer.Argument(
        ...,
        help="Path to config.json, model directory, or Hugging Face model ID",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the migrated config to a file instead of stdout",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        help="Override dataset name instead of autodetecting it from the config",
    ),
):
    """
    Migrate a model config to the curated dataset channel schema.
    """
    config_path, source_name = resolve_config_source(source)
    cfg = json.loads(config_path.read_text())
    source_names = [source_name]
    source_path = Path(source_name)
    if source_path.name:
        source_names.append(source_path.name)
    cfg["channels"] = resolve_dataset_channels(
        cfg,
        dataset=dataset,
        source_names=source_names,
    )

    rendered = json.dumps(cfg, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(rendered + "\n")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    cli()
