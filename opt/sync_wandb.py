import json
import re
from pathlib import Path
from typing import Optional
import logging

import wandb

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
    else:
        return hostname


def run_summary(run):
    config = run.config
    stats = {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "tags": run.tags,
        "hostname": get_cluster(run.metadata["host"]),
        "created": run.metadata["startedAt"],
        "steps": run.summary["trainer/global_step"],
        "lr": get_entry(config, "cli", "model", "optimizer", "lr"),
        "optimizer": get_entry(config, "cli", "model", "optimizer", "class_path"),
        "tokenizer": get_entry(config, "cli", "data", "tokenizer")
        or get_entry(config, "tokenizer"),
        "num_training_steps": get_entry(
            config, "cli", "model", "lr_schedule", "num_training_steps"
        ),
        "macro_batch_size": get_entry(config, "stats/train_macro_batch_size"),
        "gas": get_entry(config, "cli", "trainer", "accumulate_grad_batches") or 1,
    }

    # Get validation loss
    val_loss = run.summary["val/loss_epoch"]
    if isinstance(val_loss, float):
        stats["val_loss_last"] = val_loss
    else:
        stats["val_loss_best"] = val_loss["min"]

    stats["effective_batch_size"] = stats["macro_batch_size"] * stats["gas"]
    return stats


def pretraining_summary(run):
    config = run.config
    row = run_summary(run)
    row.update(
        {
            "d_ff": get_entry(config, "cli", "model", "intermediate_size")
            or get_entry(config, "intermediate_size"),
            "d_model": get_entry(config, "cli", "model", "hidden_size")
            or get_entry(config, "hidden_size"),
            "n_layers": get_entry(config, "cli", "model", "num_hidden_layers")
            or get_entry(config, "num_hidden_layers"),
            "n_heads": get_entry(config, "cli", "model", "num_attention_heads")
            or get_entry(config, "num_attention_heads"),
            "lr": get_entry(config, "cli", "model", "optimizer", "lr"),
            "optimizer": get_entry(config, "cli", "model", "optimizer", "class_path"),
            "dataset": get_entry(config, "cli", "data", "path")
            or get_entry(config, "path"),
        }
    )
    row["model_size"] = model_size(row["d_model"], row["d_ff"], row["n_layers"])
    return row


METRIC_REGEX = re.compile(r"(?P<split>\w+)/(?P<metric>[a-zA-Z0-9]+)_(?P<tok_group>\w+)")


def finetuning_summary(run):
    row = run_summary(run)
    config = run.config
    row.update(
        {
            "encoder_ckpt": get_entry(config, "cli", "model", "encoder_ckpt"),
            "dataset": get_entry(config, "cli", "data", "name"),
        }
    )

    # Identify Encoder
    encoder_id = None

    if row["encoder_ckpt"].endswith("ckpt"):
        segments = Path(row["encoder_ckpt"]).parents
        id = str(segments[1].name)
        if len(id) == 8:
            encoder_id = id
    row["encoder_id"] = encoder_id

    # Identify metrics
    summary_metrics = set()
    for k in run.summary.keys():
        if m := METRIC_REGEX.match(k):
            if m.group("tok_group") in ["oov", "non_oov", "all"]:
                summary_metrics.add(m.group("metric"))

    metrics = []
    for metric in summary_metrics:
        for split in ["train", "val", "test"]:
            for tok_group in ["oov", "non_oov", "all"]:
                try:
                    metrics.append(
                        {
                            "split": split,
                            "metric": metric,
                            "token_group": tok_group,
                            "best": run.summary_metrics[
                                f"{split}/{metric}_{tok_group}"
                            ]["best"],
                        }
                    )
                except KeyError:
                    pass
    row["metrics"] = metrics
    return row


def export_runs(exportfun, cache: Path, runs, name: str = None):
    if name:
        cache = cache_path.joinpath(name)
        cache.mkdir(exist_ok=True, parents=True)
    for run in runs:
        run_cache = cache.joinpath(run.id).with_suffix(".json")
        if run_cache.exists():
            continue
        logging.info("exporting %s (%s) to %s", run.id, run.url, name or "default")
        stats = exportfun(run)
        with open(run_cache, "w") as fid:
            json.dump(stats, fid)


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
            # "summary_metrics.val/loss_epoch.min": {"$exists": True},
            "config.cli.model.init_args.encoder_ckpt": {"$exists": True},
        },
    )
    export_runs(finetuning_summary, cache_path, runs, name="finetuning")
