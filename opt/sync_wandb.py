import json
import re
from pathlib import Path
from typing import Optional, Union
import logging

import wandb
from datasets import fingerprint

logging.basicConfig(level=logging.INFO)


def model_size(d_model: int, d_ff: int, n_layers: int) -> int:
    attention_qkv = n_layers * 3 * d_model**2
    project = n_layers * d_model**2
    ff = n_layers * 2 * d_model * d_ff
    return attention_qkv + project + ff


def get_entry(config: dict, *entry_path):
    if len(entry_path) > 1:
        path = entry_path[0]
        if path in config:
            val = get_entry(config[path], *entry_path[1:])
            if val is not None:
                return val

        elif "init_args" in config:
            # print(f"init_args: {",".join(entry_path)}, {config}")
            return get_entry(config["init_args"], *entry_path)
    elif config is None:
        return None
    elif isinstance(config, str):
        return None
    else:
        # print(f"getting {entry_path} from {config}")
        val = config.get(*entry_path, None)
        if val is not None:
            return val
        elif "init_args" in config:
            return config["init_args"].get(*entry_path, None)
        return None


def get_cluster(hostname: str) -> str:
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


def summary_metric(run, key, type="last", best=None):
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


def run_summary(run):
    config = run.config
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
        },
        "job_config": {
            "nodes": get_entry(config, "job_config", "nodes"),
            "gpues_per_node": get_entry(config, "job_config", "gpus_per_node"),
            "container": get_entry(config, "job_config", "container"),
            "env": get_entry(config, "job_config", "env"),
        },
        "metrics": {
            "train_loss_last": summary_metric(run, "train/loss_step", "last"),
            "val_loss_last": summary_metric(run, "val/loss_epoch", "last"),
            "val_loss_best": summary_metric(run, "val/loss_epoch", "best", best="min"),
            "train_loss_best": summary_metric(run, "val/loss_step", "best", best="min"),
        },
    }

    # Populate Effective Batch Size
    stats["trainer"]["effective_batch_size"] = (
        stats["trainer"]["macro_batch_size"] * stats["trainer"]["gas"]
    )
    return stats


def pretraining_summary(run):
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
    return row


def finetuning_summary(run):
    row = run_summary(run)
    config = run.config
    row["model"].update(
        {
            "encoder_ckpt": get_entry(config, "cli", "model", "encoder_ckpt"),
            "task": get_entry(config, "cli", "model", "task"),
        }
    )

    if row["data"]["module"] == "electrolyte_fm.data_module.tmQMDataModule":
        dataset = "tmQM"
    else:
        dataset = get_entry(config, "cli", "data", "name")

    row["data"].update({"dataset": dataset, "targets": get_entry(config, "cli", "data", "target_columns")})

    # Identify Encoder
    encoder_id = None

    if row["model"]["encoder_ckpt"].endswith("ckpt"):
        segments = Path(row["model"]["encoder_ckpt"]).parents
        id = str(segments[1].name)
        if len(id) == 8:
            encoder_id = id
    row["model"]["encoder_id"] = encoder_id

    # Identify metrics
    row["metrics"] = identify_metrics(run.summary_metrics)
    return row


METRIC_REGEX = re.compile(
    r"(?P<split>\w+)/(?P<metric>\w+?)_(?P<tok_group>(?:oov)|(?:all)|(?:non_oov))(?:_(?P<bootstrap>(?:mean)|(?:std)))?"
)


def identify_metrics(summary_metrics):
    metrics = []
    for k, v in summary_metrics.items():
        m = METRIC_REGEX.match(k)
        if m is None:
            continue

        # Unpack metric
        entry = m.groupdict()
        if isinstance(v, (float, int)):
            entry["type"] = "last"
            entry["value"] = v
            metrics.append(entry)

        elif isinstance(v, str):
            entry["type"] = "last"
            entry["value"] = v if v != "NaN" else float("nan")
            metrics.append(entry)

        else:
            # Multiple summary metrics were logged
            for sk, sv in v.items():
                metrics.append({"type": sk, "value": sv, **entry})
    return metrics


def export_runs(exportfun, cache: Path, runs, name: str = None):
    if name:
        cache = cache_path.joinpath(name)
        cache.mkdir(exist_ok=True, parents=True)
    else:
        name = "default"
    for run in runs:
        run_cache = cache.joinpath(run.id).with_suffix(".json")

        # Compute fingerprint
        fp = fingerprint.update_fingerprint(
            run.id,
            exportfun,
            {
                "run": run.id,
                "entity": run.entity,
                "project": run.project,
                "state": run.state,
            },
        )

        # Check for an existing export
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
        except TypeError:
            logging.error("failed to export %s", run.id)


if __name__ == "__main__":
    api = wandb.Api()

    # Create cache
    cache_path = Path(__file__).parent.parent.joinpath(".cache", "wandb-export")
    cache_path.mkdir(exist_ok=True, parents=True)

    # Sync Pretraining Runs
    runs = api.runs(
        "incite-mist/mist",
        filters={
            "State": {"$in": ["Crashed", "Finished"]},
            "tags": {"$in": ["pretraining"], "$nin": ["debug"]},
            "summary_metrics.trainer/global_step": {"$exists": True, "$gte": 100},
            "summary_metrics.val/loss_epoch.min": {"$exists": True},
        },
    )
    export_runs(pretraining_summary, cache_path, runs, name="pretraining")

    # Sync Finetuning
    runs = api.runs(
        "incite-mist/mist",
        filters={
            "State": {"$in": ["Crashed", "Finished"]},
            "tags": {"$in": ["finetuning"], "$nin": ["debug"]},
            "summary_metrics.trainer/global_step": {"$exists": True, "$gte": 100},
            "summary_metrics.val/loss_epoch": {"$exists": True},
            "config.cli.model.init_args.encoder_ckpt": {"$exists": True},
        },
    )
    export_runs(finetuning_summary, cache_path, runs, name="finetuning")
