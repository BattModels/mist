#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "wandb",
# ]
# ///
import json
import re
import traceback
import subprocess
from pathlib import Path
from typing import Optional, Union, Mapping
from math import nan, isnan
import logging

import wandb
from wandb.apis.public import Run
from datasets import fingerprint

logging.basicConfig(level=logging.INFO)


def model_size(d_model: int, d_ff: int, n_layers: int) -> int:
    if d_model is None or d_ff is None or n_layers is None:
        return None
    attention_qkv = n_layers * 3 * d_model**2
    project = n_layers * d_model**2
    ff = n_layers * 2 * d_model * d_ff
    return attention_qkv + project + ff


def get_entry(config: dict | str | None, *entry_path: str):
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

    # Otherwise, return None
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
    value = run.summary_metrics.get(key, None)
    if value is None:
        return None

    if isinstance(value, (float, int, str)):
        return value if type == "last" else None

    # Try to get the type
    x = value.get(type, None)

    # Hail mary for loss
    if x is None and best is not None and type == "best":
        return value.get(best, None)
    return x


def system_metrics(run: Run):
    df = run.history(stream="system")
    mean = df.mean().to_dict()
    std = df.std().to_dict()
    stats = {}
    for k in mean.keys():
        stats[k] = {"mean": mean[k], "std": std[k]}

    return stats


def metric_traces(run: Run, x_axis: str, metrics: dict[str, str]) -> dict[str, list]:
    records = run.scan_history(keys=[x_axis, *metrics.keys()])
    out = {k: [] for k in ["step", *metrics.values()]}
    for sample in records:
        out["step"].append(sample[x_axis])
        for k, v in metrics.items():
            out[v].append(sample[k])

    return out


def has_hotfix(run: Run, commit: str | list[str]) -> bool:
    """Return true if the run has all of the listed commits"""
    if isinstance(commit, list):
        return all([has_hotfix(run, c) for c in commit])
    assert isinstance(commit, str)

    run_commit = run.metadata["git"]["commit"]
    assert isinstance(run_commit, str)

    # Check if run_commit has commit as an ancestor
    o = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, run_commit],
        shell=True,
        capture_output=True,
    )
    return o.returncode == 0


def run_summary(run: Run):
    config = run.config
    assert isinstance(run.metadata, dict)
    stats = {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "tags": run.tags,
        "state": run.state,
        "user": run.metadata["username"],
        "cluster": get_cluster(run.metadata["host"]),
        "hostname": run.metadata["host"],
        "created": run.metadata["startedAt"],
        "gpu": run.metadata["gpu"],
        "commit": run.metadata["git"]["commit"],
        "runtime": run.summary["_runtime"],
        "optimizer": {
            "class_path": get_entry(config, "cli", "model", "optimizer", "class_path"),
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
            "gas": get_entry(config, "cli", "trainer", "accumulate_grad_batches") or 1,
            "macro_batch_size": get_entry(config, "stats/train_macro_batch_size"),
            "step": run.summary["trainer/global_step"],
            "tokens": run.summary.get("total_tokens_step", None),
            "masked_tokens": run.summary.get("total_masked_tokens_step", None),
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
            "val_loss_best": summary_metric(run, "val/loss_epoch", "best", best="min"),
            "train_loss_best": summary_metric(run, "val/loss_step", "best", best="min"),
        },
        "system": {
            "train_throughput": summary_metric(
                run, "stats/train_batch_throughput_epoch"
            ),
            "val_throughput": summary_metric(run, "stats/val_batch_throughput", "mean"),
            "train_batch_time": summary_metric(run, "stats/train_batch_time_epoch"),
            **system_metrics(run),
        },
    }

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
        run, "trainer/global_step", {"val/loss_epoch": "val_loss"}
    )
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

    # Identify Encoder
    row["model"]["encoder_id"] = get_ckpt_id(row["model"]["encoder_ckpt"])

    # Identify metrics
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
    summary_metrics = run.summary_metrics
    metrics = get_entry(run.config, "cli", "model", "metrics")
    target_columns = get_entry(run.config, "cli", "model", "target_columns")
    for k, v in summary_metrics.items():
        m = METRIC_REGEX.match(k)
        if m is None:
            continue

        entry = m.groupdict()

        # handle MAE naming nonsense
        if metrics is not None and target_columns is not None and "mae" in metrics:
            metric = entry["metric"]
            if metric.startswith("mae") and "_" in metric and "channel" not in metric:
                # Depreciate channel naming
                entry["channel"] = entry["metric"].split("_", maxsplit=1)[1]
                if entry["channel"] in ["mean", *target_columns]:
                    entry["metric"] = "mae"
                else:
                    # Unknown channel -> Skip
                    logging.warning("Unable to parse metric %s for %s", metric, run.id)
                    continue

            elif not any([metric.startswith(c) for c in metrics]):
                # If no metric is specified, assume MAE
                entry["channel"] = entry["metric"]
                entry["metric"] = "mae"
            else:
                pass

        # Check for invalid rmse
        if entry["metric"] == "rmse" and not has_hotfix(
            run, "369d6164d74238b89c9798ce5b3c914b2501739a"
        ):
            continue  # Skip invalid rmse records

        # Unpack metric
        if isinstance(v, (float, int)):
            entry["type"] = "last"
            entry["value"] = v
            out.append(entry)

        elif isinstance(v, str):
            entry["type"] = "last"
            entry["value"] = parse_value(v)
            out.append(entry)

        else:
            # Multiple summary metrics were logged
            for sk, sv in v.items():
                out.append({"type": sk, "value": sv, **entry})
    return out


def export_runs(export_map: dict, cache: Path, runs, name: str = None):
    for run in runs:
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

        cache = cache_path.joinpath(name)
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

        # Check for an existing export
        run_cache = cache.joinpath(run.id).with_suffix(".json")
        if run_cache.exists():
            with open(run_cache, "r") as fid:
                export_fp = json.load(fid).get("fingerprint", None)

            # If the fingerprint hasn't changed, don't need to export again
            if export_fp == fp:
                logging.info(
                    "fingerprint for %s/%s matches (%s), skipping", name, run.id, fp
                )
                continue

        logging.info("exporting %s/%s (%s)", name, run.id, run.url)
        try:
            stats = exportfun(run)
            stats["fingerprint"] = fp
            with open(run_cache, "w") as fid:
                json.dump(stats, fid)
        except KeyboardInterrupt:
            raise
        except Exception:
            logging.error("failed to export %s: %s", run.id, traceback.format_exc())
            continue


if __name__ == "__main__":
    api = wandb.Api()

    # Check git history is up to date
    subprocess.run(["git", "fetch", "--all"], check=True)

    # Create cache
    cache_path = Path(__file__).parent.parent.joinpath(".cache", "wandb-export")
    cache_path.mkdir(exist_ok=True, parents=True)

    export_map = {
        "pretraining": pretraining_summary,
        "finetuning": finetuning_summary,
        "test": test_summary,
    }

    # Sync Runs
    runs = api.runs(
        "incite-mist/mist",
        filters={
            "State": {"$in": ["Crashed", "Finished"]},
            "tags": {"$in": list(export_map.keys()), "$nin": ["debug"]},
            "summary_metrics.trainer/global_step": {"$exists": True},
        },
    )
    export_runs(export_map, cache_path, runs)
