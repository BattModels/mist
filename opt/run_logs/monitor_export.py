#!/usr/bin/env python
import os
import time
import json
import re
from pathlib import Path
from collections import defaultdict
import argparse
import subprocess


def count_exported_runs(cache_dir: Path):
    """Count exported runs by category"""
    counts = {}
    for category in ["pretraining", "finetuning", "test", "default"]:
        cat_dir = cache_dir / category
        if cat_dir.exists():
            counts[category] = len(list(cat_dir.glob("*.json")))
        else:
            counts[category] = 0

    counts["total"] = sum(counts.values())
    return counts


def parse_log_file(log_path: Path):
    """Extract statistics from a log file"""
    stats = {
        "task_id": None,
        "status": "unknown",
        "exported": 0,
        "cached": 0,
        "failed": 0,
        "completed": False,
        "errors": [],
    }

    match = re.search(r"_(\d+)\.err$", str(log_path))
    if match:
        stats["task_id"] = int(match.group(1))

    try:
        with open(log_path, "r") as f:
            content = f.read()

            # Check if completed
            if "Array task" in content and "completed" in content:
                stats["completed"] = True
                stats["status"] = "completed"
            elif "ERROR" in content or "Failed" in content:
                stats["status"] = "error"

            summary_match = re.search(
                r"Summary: \{'exported': (\d+), 'cached': (\d+), 'failed': (\d+)\}",
                content,
            )
            if summary_match:
                stats["exported"] = int(summary_match.group(1))
                stats["cached"] = int(summary_match.group(2))
                stats["failed"] = int(summary_match.group(3))

            error_lines = [
                line
                for line in content.split("\n")
                if "ERROR" in line or "Failed to export" in line
            ]
            stats["errors"] = error_lines[:5]

    except Exception as e:
        stats["status"] = "error"
        stats["errors"] = [f"Failed to parse log: {e}"]

    return stats


def print_summary(cache_dir: Path, logs_dir: Path, job_id: int = None):

    print("Exported Runs:")
    counts = count_exported_runs(cache_dir)
    for category, count in counts.items():
        print(f" {category} : {count}")

    print("Launched Process Status:")
    log_files = sorted(logs_dir.glob("wandb_export_*_*.err"))

    # Aggregate statistics
    task_stats = []
    total_exported = 0
    total_cached = 0
    total_failed = 0

    for log_file in log_files:
        stats = parse_log_file(log_file)
        task_stats.append(stats)
        total_exported += stats["exported"]
        total_cached += stats["cached"]
        total_failed += stats["failed"]

    task_stats.sort(key=lambda x: x["task_id"] if x["task_id"] is not None else -1)

    completed = sum(1 for s in task_stats if s["completed"])
    print(f"total tasks : {len(task_stats)}")
    print(f"completed : {completed}/{len(task_stats)}")
    print(f"in progress : {len(task_stats) - completed}")
    print(f"exported : {total_exported}")
    print(f"cached : {total_cached}")
    print(f"failed : {total_failed}")

    if total_cached + total_exported > 0:
        progress = (
            (total_cached + total_exported)
            / (total_cached + total_exported + total_failed)
            * 100
        )
        print(f"success rate:    {progress:.1f}%")

    failed_tasks = [s for s in task_stats if s["failed"] > 0]
    if failed_tasks:
        print(f"Tasks with Failures ({len(failed_tasks)}):")
        for stat in failed_tasks:
            task_id = stat["task_id"] if stat["task_id"] is not None else "?"
            print(f"Task {task_id}: {stat['failed']} failed runs")
            if stat["errors"]:
                print(f"First error: {stat['errors'][0][:80]}...")


def watch_progress(
    cache_dir: Path, logs_dir: Path, job_id: int = None, interval: int = 30
):
    while True:
        # Clear screen
        os.system("clear" if os.name == "posix" else "cls")
        print_summary(cache_dir, logs_dir, job_id)
        time.sleep(interval)


def list_failed_runs(logs_dir: Path, output_file: Path = None):
    """Extract list of failed run IDs"""
    failed_runs = set()

    log_files = logs_dir.glob("wandb_export_*_*.err")
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()
                matches = re.findall(r"Failed to export ([a-z0-9]{8})", content)
                failed_runs.update(matches)
        except Exception:
            pass

    print(f"Found {len(failed_runs)} unique failed runs:")
    for run_id in sorted(failed_runs):
        print(f"  {run_id}")

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(sorted(failed_runs)))
        print(f"Saved to {output_file}")

    return failed_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor WandB export progress")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/wandb-export"),
        help="Cache directory (default: .cache/wandb-export)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Logs directory (default: logs)",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        help="SLURM job ID to monitor",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch progress in real-time",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval for watch mode (seconds)",
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="List all failed run IDs",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        help="Save failed run IDs to file",
    )

    args = parser.parse_args()

    if args.list_failed:
        list_failed_runs(args.logs_dir, args.failed_output)
    elif args.watch:
        watch_progress(args.cache_dir, args.logs_dir, args.job_id, args.interval)
    else:
        print_summary(args.cache_dir, args.logs_dir, args.job_id)
