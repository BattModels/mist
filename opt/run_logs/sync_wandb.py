#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "wandb",
#     "joblib",
#     "tqdm",
# ]
# ///
import os
import re
import sys
import json
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union
from math import nan, isnan
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import wandb
from wandb.apis.public import Run
from datasets import fingerprint
from joblib import Parallel, delayed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)


def model_size(d_model: int, d_ff: int, n_layers: int) -> int:
    if d_model is None or d_ff is None or n_layers is None:
        return None
    attention_qkv = n_layers * 3 * d_model**2
    project = n_layers * d_model**2
    ff = n_layers * 2 * d_model * d_ff
    return attention_qkv + project + ff


def get_entry(config: Dict | str | None, *entry_path: str):
    if isinstance(config, Mapping):
        if len(entry_path) >= 1:
            if (path := entry_path[0]) in config:
                val = get_entry(config[path], *entry_path[1:])
                if val is not None:
                    return val

            elif "init_args" in config:
                return get_entry(config["init_args"], *entry_path)
        else:
            val = config.get(entry_path, None)
            if val is not None:
                return val
            elif "init_args" in config:
                return config["init_args"].get(*entry_path, None)
            return None

    return config if len(entry_path) == 0 else None


def get_cluster(hostname: str) -> Optional[str]:
    if hostname == "localhost":
        return "h001"
    elif hostname.startswith("lh"):
        return "artemis"
    elif hostname.startswith("x"):
        return "polaris"
    elif hostname.endswith("delta.ncsa.illinois.edu"):
        return "delta"
    elif hostname.startswith("gpu"):
        return "dgx"
    else:
        return None


def summary_metric(run: Run, key, type="last", best=None):
    try:
        value = run.summary_metrics.get(key, None)
        if value is None:
            return None

        if isinstance(value, (float, int, str)):
            return value if type == "last" else None

        x = value.get(type, None)
        if x is None and best is not None and type == "best":
            return value.get(best, None)
        return x
    except (TypeError, AttributeError, KeyError) as e:
        logging.warning(f"Failed to get summary metric {key} for run: {e}")
        return None


def system_metrics(run: Run):
    try:
        df = run.history(stream="system")
        mean = df.mean().to_dict()
        std = df.std().to_dict()
        stats = {}
        for k in mean.keys():
            stats[k] = {"mean": mean[k], "std": std[k]}
        return stats
    except Exception as e:
        logging.warning(f"Failed to get system metrics: {e}")
        return {}


def metric_traces(run: Run, x_axis: str, metrics: Dict[str, str]) -> Dict[str, List]:
    """
    Extract metric traces from a WandB run.
    Allow sparse metrics (not handled by `scan_history` method).
    """
    step_records = run.scan_history(keys=[x_axis])
    steps = []
    for sample in step_records:
        step_val = sample.get(x_axis)
        if step_val is not None:
            steps.append(step_val)

    if not steps:
        logging.warning(f"No {x_axis} values found")
        return {k: [] for k in ["step", *metrics.values()]}

    out = {"step": steps}

    for wandb_key, output_key in metrics.items():
        try:
            records = run.scan_history(keys=[x_axis, wandb_key])
            metric_dict = {}
            for sample in records:
                step_val = sample.get(x_axis)
                metric_val = sample.get(wandb_key)
                if step_val is not None:
                    metric_dict[step_val] = metric_val

            out[output_key] = [metric_dict.get(s) for s in steps]

        except Exception as e:
            logging.warning(f"Failed to get metric '{wandb_key}': {e}")
            out[output_key] = [None] * len(steps)

    return out


def has_hotfix(run: Run, commit: str | List[str]) -> bool:
    """Return true if the run has all of the listed commits"""
    try:
        if isinstance(commit, list):
            return all([has_hotfix(run, c) for c in commit])
        assert isinstance(commit, str)

        run_commit = run.metadata["git"]["commit"]
        assert isinstance(run_commit, str)

        o = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, run_commit],
            capture_output=True,
            text=True,
        )
        return o.returncode == 0
    except Exception:
        return False


def run_summary(run: Run):
    config = run.config

    try:
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        git_info = metadata.get("git", {})
        summary = run.summary if isinstance(run.summary, dict) else {}
        summary_metrics = (
            dict(run.summary_metrics)
            if hasattr(run, "summary_metrics") and run.summary_metrics
            else {}
        )
        sys_metrics = system_metrics(run)

        stats = {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "tags": run.tags,
            "state": run.state,
            "user": metadata.get("username"),
            "cluster": get_cluster(metadata.get("host", "")),
            "hostname": metadata.get("host"),
            "created": metadata.get("startedAt"),
            "gpu": metadata.get("gpu"),
            "commit": git_info.get("commit") if isinstance(git_info, dict) else None,
            "runtime": summary.get("_runtime")
            or summary_metrics.get("_runtime")
            or get_entry(summary_metrics, "_wandb", "runtime"),
            "optimizer": {
                "class_path": get_entry(
                    config, "cli", "model", "optimizer", "class_path"
                ),
                "lr": get_entry(config, "cli", "model", "optimizer", "lr"),
                "betas": get_entry(config, "cli", "model", "optimizer", "betas"),
            },
            "model": {
                "class_path": get_entry(config, "cli", "model", "class_path"),
            },
            "data": {
                "module": get_entry(config, "cli", "data", "class_path"),
                "tokenizer": get_entry(config, "cli", "data", "tokenizer"),
                "batch_size": get_entry(config, "cli", "data", "batch_size"),
                "encoding": get_entry(config, "cli", "data", "encoding"),
            },
            "trainer": {
                "num_training_steps": get_entry(
                    config, "cli", "model", "lr_schedule", "num_training_steps"
                ),
                "gas": get_entry(config, "cli", "trainer", "accumulate_grad_batches")
                or 1,
                "macro_batch_size": get_entry(config, "stats/train_macro_batch_size"),
                "step": summary.get("trainer/global_step"),
                "tokens": summary.get("total_tokens_step"),
                "masked_tokens": summary.get("total_masked_tokens_step"),
                "limit_val_batches": get_entry(
                    config, "cli", "trainer", "limit_val_batches"
                ),
            },
            "job_config": {
                "nodes": get_entry(config, "n_nodes"),
                "gpus_per_node": get_entry(config, "n_gpus_per_node"),
                "container": get_entry(config, "job_config", "container"),
                "env": get_entry(config, "job_config", "env"),
            },
            "metrics": {
                "train_loss_last": summary_metric(run, "train/loss_step", "last"),
                "val_loss_last": summary_metric(run, "val/loss_epoch", "last"),
                "val_loss_best": summary_metric(
                    run, "val/loss_epoch", "best", best="min"
                ),
                "train_loss_best": summary_metric(
                    run, "val/loss_step", "best", best="min"
                ),
            },
            "system": {
                "train_throughput": summary_metric(
                    run, "stats/train_batch_throughput_epoch"
                ),
                "val_throughput": summary_metric(
                    run, "stats/val_batch_throughput", "mean"
                ),
                "train_batch_time": summary_metric(run, "stats/train_batch_time_epoch"),
                **sys_metrics,
            },
            "summary_metrics": summary_metrics,
        }
        print(stats["id"], stats["runtime"])
        # Record world_size
        world_size = (stats["job_config"]["nodes"] or nan) * (
            stats["job_config"]["gpus_per_node"] or nan
        )
        stats["job_config"]["world_size"] = nan2none(world_size)

        # Populate Effective Batch Size
        macro_batch_size = stats["trainer"]["macro_batch_size"]
        stats["data"]["val_batch_size"] = (
            get_entry(config, "cli", "data", "val_batch_size")
            or stats["data"]["batch_size"]
        )
        gas = stats["trainer"]["gas"]
        if macro_batch_size is not None and gas is not None:
            stats["trainer"]["effective_batch_size"] = macro_batch_size * gas
        else:
            stats["trainer"]["effective_batch_size"] = None

        limit_val_batches = stats["trainer"]["limit_val_batches"] or nan
        val_batch_size = stats["data"]["val_batch_size"] or nan
        stats["trainer"]["effective_val_epoch"] = nan2none(
            limit_val_batches * val_batch_size * world_size
        )

        return stats
    except Exception as e:
        logging.error(f"Error in run_summary for {run.id}: {e}")
        raise


def pretraining_summary(run: Run):
    config = run.config
    row = run_summary(run)
    row["model"].update(
        {
            "d_ff": get_entry(config, "cli", "model", "intermediate_size"),
            "d_model": get_entry(config, "cli", "model", "hidden_size"),
            "n_layers": get_entry(config, "cli", "model", "num_hidden_layers"),
            "n_heads": get_entry(config, "cli", "model", "num_attention_heads"),
        }
    )
    row["data"].update({"path": get_entry(config, "cli", "data", "path")})
    row["model"]["model_size"] = model_size(
        row["model"]["d_model"], row["model"]["d_ff"], row["model"]["n_layers"]
    )
    row["metric_traces"] = metric_traces(
        run,
        "trainer/global_step",
        {"val/loss_epoch": "val_loss", "total_tokens_step": "total_tokens_step"},
    )
    row["metrics"] = identify_metrics(run)
    return row


def identify_dataset(config):
    if (
        get_entry(config, "cli", "data", "class_path")
        == "electrolyte_fm.data_modules.tmQMDataModule"
    ):
        path = get_entry(config, "cli", "data", "path")
        if path is None:
            return "tmQM"
        if path.endswith("data_tm_split"):
            return "tmQM-tm-split"
        else:
            return "tmQM"
    else:
        return get_entry(config, "cli", "data", "name")


def get_ckpt_id(ckpt):
    if ckpt is None:
        return None
    if not ckpt.endswith("ckpt"):
        return None

    segments = Path(ckpt).parents
    id = str(segments[1].name)
    if len(id) == 8:
        return id
    return None


def something(x, default):
    """Return x if not None, otherwise default"""
    return x if x is not None else default


def nan2none(x):
    return x if not isnan(x) else None


def finetuning_summary(run: Run):
    row = run_summary(run)
    config = run.config
    row["model"].update(
        {
            "encoder_ckpt": get_entry(config, "cli", "model", "encoder_ckpt"),
            "task": get_entry(config, "cli", "model", "task"),
            "metrics": get_entry(config, "cli", "model", "metrics"),
            "freeze_encoder": something(
                get_entry(config, "cli", "model", "freeze_encoder"), True
            ),
        }
    )

    row["data"].update(
        {
            "dataset": identify_dataset(config),
            "targets": get_entry(config, "cli", "data", "target_columns"),
        }
    )
    row["metric_traces"] = metric_traces(
        run,
        "trainer/global_step",
        {"val/loss_epoch": "val_loss", "total_tokens_step": "total_tokens_step"},
    )
    row["model"]["encoder_id"] = get_ckpt_id(row["model"]["encoder_ckpt"])
    row["metrics"] = identify_metrics(run)
    return row


def test_summary(run):
    row = run_summary(run)
    config = run.config
    row["model"].update(
        {
            "ckpt": get_entry(config, "cli", "ckpt_path"),
            "task": get_entry(config, "cli", "model", "task"),
            "ckpt_id": get_ckpt_id(get_entry(config, "cli", "ckpt_path")),
            "metrics": get_entry(config, "cli", "model", "metrics"),
        }
    )
    row["data"].update(
        {
            "dataset": identify_dataset(config),
            "targets": get_entry(config, "cli", "data", "target_columns"),
        }
    )
    row["metrics"] = identify_metrics(run)
    return row


METRIC_REGEX = re.compile(
    r"^(?P<split>\w+)/(?P<metric>.+?)(:?_channel_(?P<channel>.+?))?(?:_(?P<bootstrap>mean|std))?(?:_(?P<tok_group>all|oov|non_oov))?$"
)


def parse_value(v: str) -> Union[float, str]:
    if v == "NaN":
        return float("nan")
    elif v == "Infinity":
        return float("inf")
    else:
        return v


def identify_metrics(run):
    out = []
    try:
        summary_metrics = run.summary_metrics
        metrics = get_entry(run.config, "cli", "model", "metrics")
        target_columns = get_entry(run.config, "cli", "model", "target_columns")

        for k, v in summary_metrics.items():
            m = METRIC_REGEX.match(k)
            if m is None:
                continue

            entry = m.groupdict()

            if metrics is not None and target_columns is not None and "mae" in metrics:
                metric = entry["metric"]
                if (
                    metric.startswith("mae")
                    and "_" in metric
                    and "channel" not in metric
                ):
                    entry["channel"] = entry["metric"].split("_", maxsplit=1)[1]
                    if entry["channel"] in ["mean", *target_columns]:
                        entry["metric"] = "mae"
                    else:
                        logging.warning(
                            "Unable to parse metric %s for %s", metric, run.id
                        )
                        continue

                elif not any([metric.startswith(c) for c in metrics]):
                    entry["channel"] = entry["metric"]
                    entry["metric"] = "mae"
                else:
                    pass

            if entry["metric"] == "rmse" and not has_hotfix(
                run, "369d6164d74238b89c9798ce5b3c914b2501739a"
            ):
                continue

            if isinstance(v, (float, int)):
                entry["type"] = "last"
                entry["value"] = v
                out.append(entry)

            elif isinstance(v, str):
                entry["type"] = "last"
                entry["value"] = parse_value(v)
                out.append(entry)

            else:
                for sk, sv in v.items():
                    out.append({"type": sk, "value": sv, **entry})
    except Exception as e:
        logging.warning(f"Failed to identify metrics for run {run.id}: {e}")

    return out


def process_single_run(
    run_id: str, entity: str, project: str, export_map: Dict, cache_base: Path
):
    try:
        api = wandb.Api(timeout=120)
        run = api.run(f"{entity}/{project}/{run_id}")

        # Clean up tags if needed
        if "pretraining" in run.tags and "finetuning" in run.tags:
            run.tags.remove("pretraining")
            run.update()

        # Identify export function
        exportfun = run_summary
        name = "default"
        for tag, f in export_map.items():
            if tag in run.tags:
                exportfun = f
                name = tag

        cache = cache_base.joinpath(name)
        cache.mkdir(exist_ok=True, parents=True)

        # Compute fingerprint
        fp = fingerprint.update_fingerprint(
            run.id,
            exportfun,
            {
                "run": run.id,
                "entity": run.entity,
                "project": run.project,
                "state": run.state,
                "tags": run.tags,
            },
        )

        # Check for existing export
        run_cache = cache.joinpath(run.id).with_suffix(".json")
        if run_cache.exists():
            with open(run_cache, "r") as fid:
                export_fp = json.load(fid).get("fingerprint", None)

            if export_fp == fp:
                return {"status": "cached", "run_id": run.id, "name": name}

        # Export the run
        stats = exportfun(run)
        stats["fingerprint"] = fp

        with open(run_cache, "w") as fid:
            json.dump(stats, fid)

        return {"status": "exported", "run_id": run.id, "name": name}

    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = traceback.format_exc()
        logging.error(f"Failed to export {run_id}: {error_msg}")
        return {"status": "failed", "run_id": run_id, "error": str(e)}


def export_runs(
    export_map: Dict, cache: Path, runs, n_jobs: int = -1, batch_size: int = None
):
    # Extract run IDs and metadata
    run_info = [(r.id, r.entity, r.project) for r in runs]

    logging.info(f"Processing {len(run_info)} runs with {n_jobs} workers")

    results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
        delayed(process_single_run)(run_id, entity, project, export_map, cache)
        for run_id, entity, project in run_info
    )

    # Summarize results
    summary = {"exported": 0, "cached": 0, "failed": 0}
    failed_runs = []

    for result in results:
        status = result["status"]
        summary[status] += 1
        if status == "failed":
            failed_runs.append((result["run_id"], result.get("error", "Unknown")))

    logging.info(f"Summary: {summary}")
    if failed_runs:
        logging.warning(f"Failed runs ({len(failed_runs)}):")
        for run_id, error in failed_runs[:10]:  # Show first 10
            logging.warning(f"  {run_id}: {error}")

    return summary


def chunk_runs_slurm_array(runs, array_id: int, array_size: int):
    run_list = list(runs)
    chunk_size = len(run_list) // array_size + (1 if len(run_list) % array_size else 0)
    start_idx = array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(run_list))
    return run_list[start_idx:end_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export WandB runs in parallel")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 for all CPUs)",
    )
    parser.add_argument(
        "--slurm-array-id",
        type=int,
        default=None,
        help="SLURM array task ID (for array job mode)",
    )
    parser.add_argument(
        "--slurm-array-size",
        type=int,
        default=None,
        help="Total number of SLURM array tasks",
    )
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument(
        "--entity", type=str, default="incite-mist", help="WandB entity"
    )
    parser.add_argument("--project", type=str, default="mist", help="WandB project")

    args = parser.parse_args()

    api = wandb.Api(timeout=120)

    # Fetch git history
    try:
        subprocess.run(["git", "fetch", "--all"], check=True, timeout=120)
    except Exception as e:
        logging.warning(f"Failed to fetch git history: {e}")

    # Create cache
    cache_path = Path(__file__).parent.parent.parent.joinpath(".cache", "wandb-export")
    cache_path.mkdir(exist_ok=True, parents=True)

    export_map = {
        "pretraining": pretraining_summary,
        "finetuning": finetuning_summary,
        "test": test_summary,
    }

    # Fetch runs
    logging.info("Fetching runs from WandB...")
    runs = api.runs(
        f"{args.entity}/{args.project}",
        filters={
            "State": {"$in": ["Crashed", "Finished"]},
            "tags": {"$in": list(export_map.keys()), "$nin": ["debug"]},
            "summary_metrics.trainer/global_step": {"$exists": True},
        },
    )

    runs_list = list(runs)
    logging.info(f"Found {len(runs_list)} runs to process")

    # Handle SLURM array mode
    if args.slurm_array_id is not None and args.slurm_array_size is not None:
        logging.info(
            f"Running in SLURM array mode: task {args.slurm_array_id}/{args.slurm_array_size}"
        )
        runs_list = chunk_runs_slurm_array(
            runs_list, args.slurm_array_id, args.slurm_array_size
        )
        logging.info(f"Processing {len(runs_list)} runs in this task")

    # Export runs
    if args.sequential:
        logging.info("Running in sequential mode")
        for run in tqdm(runs_list):
            process_single_run(run.id, run.entity, run.project, export_map, cache_path)
    else:
        export_runs(export_map, cache_path, runs_list, n_jobs=args.n_jobs)
