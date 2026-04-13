#!/usr/bin/env python

from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Tuple

import typer
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

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


def iter_org_model_repo_ids(api: HfApi, org: str) -> Iterable[str]:
    for model in api.list_models(author=org):
        repo_id = getattr(model, "modelId", None) or getattr(model, "id", None)
        if repo_id:
            yield repo_id


def render_migrated_config(
    cfg: dict,
    *,
    dataset: Optional[str],
    source_name: str,
) -> str:
    source_names = [source_name]
    source_path = Path(source_name)
    if source_path.name:
        source_names.append(source_path.name)
    cfg["channels"] = resolve_dataset_channels(
        cfg,
        dataset=dataset,
        source_names=source_names,
    )
    return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"


def normalize_repo_id(org: str, repo: str) -> str:
    return repo if "/" in repo else f"{org}/{repo}"


def dry_run_output_path(repo_id: str) -> Path:
    return Path.cwd() / repo_id.split("/")[-1] / "config.json"


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
    rendered = render_migrated_config(cfg, dataset=dataset, source_name=source_name)
    if output is not None:
        output.write_text(rendered)
    else:
        typer.echo(rendered)


@cli.command("org")
def migrate_org(
    org: str = typer.Option(..., help="HF organization name (namespace)"),
    repo: Optional[str] = typer.Option(
        None,
        help="Optional repo name or full repo id to migrate just one model",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        help="Override dataset name instead of autodetecting it from the config",
    ),
    revision: str = typer.Option(
        "main", help='Base revision/branch to update (default: "main")'
    ),
    commit_message: str = typer.Option(
        "Migrate channel config", help="Commit title / PR title"
    ),
    pr: bool = typer.Option(True, help="Create a PR instead of commiting directly"),
    dry_run: bool = typer.Option(False, help="Print actions, do not create PRs"),
    skip_fail: bool = typer.Option(
        True,
        help="Skip repos when dataset resolution fails instead of stopping",
    ),
):
    """
    Migrate config.json across model repos in an org and open PRs for the changes.
    """
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if repo is not None:
        repo_ids = [normalize_repo_id(org, repo)]
    else:
        repo_ids = list(iter_org_model_repo_ids(api, org))

    typer.echo(f"Found {len(repo_ids)} model repos to inspect")

    for repo_id in repo_ids:
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                repo_type="model",
                revision=revision,
                token=token,
            )
        except Exception:
            typer.echo(f"Failed to download {repo_id}")

        existing = Path(config_path).read_text()
        cfg = json.loads(existing)
        try:
            rendered = render_migrated_config(cfg, dataset=dataset, source_name=repo_id)
        except Exception as exc:
            if skip_fail:
                typer.echo(f"Skipping {repo_id}: {exc}")
                continue
            raise

        if rendered == existing:
            typer.echo(f"Skipping {repo_id}: config.json already up to date")
            continue

        if dry_run:
            output_path = dry_run_output_path(repo_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            typer.echo(f"[DRY RUN] Wrote {output_path}")
            continue

        op = CommitOperationAdd(
            path_in_repo="config.json",
            path_or_fileobj=BytesIO(rendered.encode("utf-8")),
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            operations=[op],
            commit_message=commit_message,
            create_pr=pr,
        )
        if pr:
            typer.echo(f"Opened PR for {repo_id}")
        else:
            typer.echo(f"Migrated config for {repo_id}")


if __name__ == "__main__":
    cli()
