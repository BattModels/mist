#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "typer",
#     "rich",
# ]
# ///
"""
Query WandB runs from export.
"""
from pathlib import Path
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=True, no_args_is_help=True)

DEFAULT_CACHE = Path("run-logs")
DEFAULT_OUT = Path(".")
PRODUCTION_MODELS: Dict[str, str] = {"MIST-1.8B": "dh61satt", "MIST-28M": "ti624ev1"}

console = Console()


def _max_num(seq: Any) -> Optional[float]:
    if isinstance(seq, (list, tuple)):
        nums = [x for x in seq if isinstance(x, (int, float))]
        return max(nums) if nums else None
    return float(seq) if isinstance(seq, (int, float)) else None


def _min_num(seq: Any) -> Optional[float]:
    if isinstance(seq, (list, tuple)):
        nums = [x for x in seq if isinstance(x, (int, float))]
        return min(nums) if nums else None
    return float(seq) if isinstance(seq, (int, float)) else None


def impute_step(data: Dict[str, Any]) -> Optional[int]:

    t = data.get("trainer")
    if isinstance(t, dict) and isinstance(t.get("step"), (int, float)):
        return int(t["step"])

    t = data.get("summary_metrics")
    if isinstance(t, dict) and isinstance(t.get("_step"), (int, float)):
        return int(t["_step"])

    traces = data.get("metric_traces")
    mv = _max_num(traces.get("step"))
    if isinstance(mv, (int, float)):
        return int(mv)
    return None


def impute_tokens(data: Dict[str, Any]) -> Optional[float]:

    t = data.get("trainer")

    if isinstance(t, dict) and isinstance(t.get("tokens", None), int):
        return t.get("tokens")

    s = data.get("summary_metrics")
    if isinstance(s, dict) and isinstance(s.get("total_tokens_step", None), int):
        return s.get("total_tokens_step")

    traces = data.get("metric_traces")
    if isinstance(traces, dict):
        return _max_num(traces.get("total_tokens_step"))

    return None


def extract_val_loss(data: Dict[str, Any]) -> Optional[float]:
    s = data.get("summary_metrics")

    if isinstance(s, dict):
        val_loss_mn = s.get("val/loss_step", {})
        if isinstance(val_loss_mn, dict):
            val_loss_mn = val_loss_mn.get("min", None)
            if isinstance(val_loss_mn, float):
                return val_loss_mn

    traces = data.get("metric_traces", {})
    if isinstance(traces.get("val_loss"), list):
        return _min_num(traces["val_loss"])
    return None


def rich_table(
    df: pl.DataFrame,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> None:

    if limit is not None:
        df = df.head(limit)
    cols = [c for c in (columns or df.columns) if c in df.columns]
    table = Table(show_header=True)
    for col in cols:
        table.add_column(col, max_width=22, no_wrap=True, overflow="ellipsis")
    for row in df.select(cols).iter_rows():
        table.add_row(
            *[
                "" if v is None else (f"{v:.5g}" if isinstance(v, float) else str(v))
                for v in row
            ]
        )
    console.print(table)


def flatten_run(data: Dict[str, Any], run_type: str) -> Dict[str, Any]:
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    system = data.get("system") if isinstance(data.get("system"), dict) else {}
    data_cfg = data.get("data") if isinstance(data.get("data"), dict) else {}

    flat: Dict[str, Any] = {
        "id": data.get("id"),
        "name": data.get("name"),
        "runtime": data.get("runtime"),
        "tags": ",".join(data.get("tags", []) or []),
        "tokenizer": data_cfg.get("tokenizer"),
        "batch_size": data_cfg.get("batch_size"),
        "full_data": json.dumps(data, separators=(",", ":"), ensure_ascii=False),
    }

    if run_type == "pretraining":
        flat.update(
            {
                "d_model": model.get("d_model"),
                "d_ff": model.get("d_ff"),
                "n_layers": model.get("n_layers"),
                "n_heads": model.get("n_heads"),
                "model_size": model.get("model_size"),
                "step": impute_step(data),
                "tokens": impute_tokens(data),
                "val_loss_last": extract_val_loss(data),
                "val_loss_best": extract_val_loss(data),
                "train_throughput": system.get("train_throughput"),
            }
        )
    elif run_type == "finetuning":
        targets = data_cfg.get("targets")
        if isinstance(targets, list):
            targets = ",".join(str(t) for t in targets)
        flat.update(
            {
                "encoder_id": model.get("encoder_id"),
                "task": model.get("task"),
                "freeze_encoder": bool(model.get("freeze_encoder")),
                "dataset": data_cfg.get("dataset"),
                "targets": targets,
                "step": impute_step(data),
                "val_loss": extract_val_loss(data),
            }
        )
    return flat


def load_runs(cache_dir: Path, run_type: str) -> pl.DataFrame:
    p = cache_dir / run_type
    if not p.exists():
        return pl.DataFrame()
    rows: List[Dict[str, Any]] = []
    for jf in sorted(p.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        rows.append(flatten_run(d, run_type))
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def create_database(cache_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_type in ("pretraining", "finetuning", "test"):
        df = load_runs(cache_dir, run_type)
        if not df.is_empty():
            out = output_dir / f"{run_type}.parquet"
            df.write_parquet(out, compression="snappy")
            logging.info("%s: %d → %s", run_type, df.height, out)
        else:
            logging.info("%s: 0", run_type)


def get_production_moleculenet(parquet_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)

    filtered = df.with_columns(pl.col("tags").fill_null("")).filter(
        pl.col("tags").str.contains("finetuning")
        & pl.col("encoder_id").is_in(list(PRODUCTION_MODELS.values()))
    )

    mins = (
        filtered.with_columns(
            pl.col("val_loss").fill_null(float("inf")).alias("val_loss_filled")
        )
        .group_by(["dataset", "targets", "encoder_id", "freeze_encoder"])
        .agg(pl.col("val_loss_filled").min().alias("val_loss_min"))
    )

    best_rows = (
        filtered.with_columns(
            pl.col("val_loss").fill_null(float("inf")).alias("val_loss_filled")
        )
        .join(
            mins, on=["dataset", "targets", "encoder_id", "freeze_encoder"], how="inner"
        )
        .filter(pl.col("val_loss_filled") == pl.col("val_loss_min"))
        .group_by(["dataset", "targets", "encoder_id", "freeze_encoder"])
        .head(1)
    )

    rows: List[Dict[str, Any]] = []
    for r in best_rows.iter_rows(named=True):
        full = json.loads(r["full_data"])
        model = full.get("model")
        optim = full.get("optimizer")
        data_cfg = full.get("data")

        rows.append(
            {
                "dataset": r["dataset"],
                "targets": r["targets"],
                "encoder_id": r["encoder_id"],
                "freeze_encoder": bool(model.get("freeze_encoder")),
                "task": model.get("task"),
                "lr": optim.get("lr"),
                "batch_size": data_cfg.get("batch_size"),
                "step": impute_step(full),
            }
        )

    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["dataset", "targets", "encoder_id", "freeze_encoder"]
    )


def get_production_pretraining(cache_dir: Path) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    pre_dir = cache_dir / "pretraining"
    for model_name, run_id in PRODUCTION_MODELS.items():
        jp = pre_dir / f"{run_id}.json"
        if not jp.exists():
            logging.warning("missing: %s (%s)", model_name, jp)
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        model = d.get("model")
        tokens = impute_tokens(d)
        rows.append(
            {
                "model": model_name,
                "params_M": model.get("model_size") / 1e6,
                "d_model": model.get("d_model"),
                "n_layers": model.get("n_layers"),
                "n_heads": model.get("n_heads"),
                "step": impute_step(d),
                "tokens_B": tokens / 1e9,
                "val_loss_best": extract_val_loss(d),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


@app.command()
def build(
    cache_dir: Path = typer.Option(DEFAULT_CACHE, "--cache-dir", "-c"),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output-dir", "-o"),
) -> None:
    create_database(cache_dir, output_dir)


@app.command()
def summary(
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output-dir", "-o"),
) -> None:
    for name in ("pretraining", "finetuning", "test"):
        p = output_dir / f"{name}.parquet"
        if p.exists():
            n = pl.read_parquet(p).height
            logging.info("%s: %d (%s)", name, n, p)
        else:
            logging.info("%s: 0 (%s)", name, p)


@app.command()
def production_finetuning(
    parquet: Optional[Path] = typer.Option(None, "--parquet"),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output-dir", "-o"),
    csv: Optional[Path] = typer.Option(None, "--csv"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    parquet_file = parquet or (output_dir / "finetuning.parquet")
    if not parquet_file.exists():
        logging.error("%s not found; run build first or pass --parquet", parquet_file)
        raise typer.Exit(1)
    df = get_production_moleculenet(parquet_file)
    rich_table(df, limit=limit)
    if csv:
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(csv)
        logging.info("saved: %s", csv)


@app.command()
def production_pretraining(
    cache_dir: Path = typer.Option(DEFAULT_CACHE, "--cache-dir", "-c"),
    csv: Optional[Path] = typer.Option(None, "--csv"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    df = get_production_pretraining(cache_dir)
    cols = [
        "model",
        "params_M",
        "d_model",
        "n_layers",
        "n_heads",
        "step",
        "tokens_B",
        "val_loss_best",
        "runtime_h",
    ]
    cols = [c for c in cols if c in df.columns]
    rich_table(df, columns=cols, limit=limit)
    if csv:
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(csv)
        logging.info("saved: %s", csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
