#!/usr/bin/env python3
import os
import subprocess
import asyncio
import logging
import argparse
from uuid import uuid4
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def get_node_list():
    nodes = []
    with open(os.environ["PBS_NODEFILE"], "r") as fid:
        for line in fid:
            nodes.append(line.strip)

    return nodes


async def launch_job(config: str, nodes: list[str], gpus_per_node: int = 4):
    np = len(nodes) * gpus_per_node

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    worker_dir = tmpdir.joinpath(uuid4())
    worker_dir.mkdir()
    with open(worker_dir.joinpath("hostfile"), "w") as fid:
        for node in nodes:
            fid.write(node + "\n")

    with open(worker_dir.joinpath("lightning.json"), "w") as fid:
        fid.write(config)

    # Copy worker_dir to node
    for node in nodes:
        subprocess.run(["-r", worker_dir, node + ":" + worker_dir], "scp")

    # Launch the worker
    proc = await asyncio.create_subprocess_exec(
        "mpiexec",
        [
            f"--np={np:d}",
            f"--ppn={gpus_per_node:d}",
            f"--hostfile={worker_dir.joinpath('hostfile')}",
            "--cpu-bind=numa",
            "python3",
            "train.py",
            "fit",
            f"--config={worker_dir.joinpath('lightning.json')}",
        ],
    )
    await proc.wait()


async def worker(node: str, queue):
    while True:
        job_config = await queue.get()
        logging.info("%s launching job %d", node)
        await launch_job(job_config, [node], gpus_per_node=4)
        logging.info("%s finished job", node)


async def scheduler(jobs: list[str], nodes: list[str]):
    logging.info("populating queue with %d items", len(jobs))
    queue = asyncio.Queue(maxsize=0)
    for job in jobs:
        queue.put_nowait(job)

    tasks = []
    logging.info("starting %d worker", len(nodes))
    for node in nodes:
        task = asyncio.create_task(worker(node, queue))
        tasks.append(task)

    # Wait for work to complete
    await queue.join()
    logging.info("queue complete, shutting down workers")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=str)

    args = parser.parse_args()
    jobs = []
    with open(args.jobs, "r") as fid:
        for line in fid:
            jobs.append(line.strip())

    scheduler(jobs, get_node_list())
