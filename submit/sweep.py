#!/usr/bin/env python3
import argparse
import json
import linecache
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from shutil import which
from socket import getfqdn
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, Callable, Optional
from uuid import uuid4

logging.basicConfig(level=logging.DEBUG)


def get_node_list():
    nodes = []
    with open(os.environ["PBS_NODEFILE"], "r") as fid:
        for line in fid:
            nodes.append(line.strip)

    return nodes


def launch(sweep_path: Path, idx: int):
    sweep_path = Path(sweep_path)
    logging.info("Starting job %d from %s", idx, sweep_path)
    with NamedTemporaryFile(suffix=f"{hash(sweep_path.name)}-{idx}.json") as pl_config:
        with open(pl_config.name, "w") as fid:
            config = json.loads(linecache.getline(str(sweep_path), idx))
            fit_config = {"fit": config["train"]}
            json.dump(fit_config, fid)

        logging.info("Dumped config to %s: %s", pl_config.name, fit_config)

        # Path up slurm environment for lightning
        env = os.environ.copy()
        if ntasks := env.get("SLURM_STEP_NUM_TASKS"):
            env["SLURM_NTASKS"] = ntasks
            env["SLURM_NTASKS_PER_NODE"] = env["SLURM_STEP_TASKS_PER_NODE"]
            env["SLURM_NNODES"] = env["SLURM_STEP_NUM_NODES"]

        subprocess.run(
            [
                "submit/set_node_rank",
                sys.executable,
                "train.py",
                "--config",
                pl_config.name,
            ],
            check=True,
            env=env,
        )


@dataclass
class JobConfig:
    lineno: int
    config: str
    gpus_per_node: int = field(init=False)
    nodes: int = field(init=False)

    def __post_init__(self):
        config = json.loads(self.config)
        self.gpus_per_node = config["gpus_per_node"]
        self.nodes = config["nodes"]


@dataclass
class Worker:
    name: Any
    launch: Callable[[JobConfig], subprocess.Popen]

    def __hash__(self):
        return hash(self.name)


def srun(job: JobConfig, sweep_path: Path, worker_id: int):
    logging.info("Running job %d: %s", job.lineno, job.config)
    job_config = json.loads(job.config)
    container = job_config.get(
        "container", "/lustre/fs0/awadell/sqsh-files/0535844560745234+mist+latest.sqsh"
    )
    pwd = os.getcwd()
    srun_args = [
        "--exact",
        f"--job-name=worker-{worker_id}",
        f"--nodes={job.nodes:d}",
        f"--ntasks={job.nodes * job.gpus_per_node:d}",
        f"--gpus-per-node={job.gpus_per_node:d}",
        f"--container-image={container}",
        "--container-mounts=/lustre/fs0,/tmp",
        f"--container-workdir={pwd}",
        "--container-readonly",
        f"--output=slurm-%j.{job.lineno}.out",
        "--open-mode=append",
        "--export=ALL",
        "/mist/.venv/bin/python3",
        __file__,
        "launch",
        str(sweep_path),
        str(job.lineno),
    ]
    logging.info("srun: %s", srun_args)
    return subprocess.Popen(srun_args, executable=which("srun"))


def qsub(job: JobConfig, nodes: list[str]):
    np = len(nodes) * job.gpus_per_node
    assert len(nodes) == job.nodes

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    worker_dir = tmpdir.joinpath(uuid4())
    worker_dir.mkdir()
    with open(worker_dir.joinpath("hostfile"), "w") as fid:
        for node in nodes:
            fid.write(node + "\n")

    with open(worker_dir.joinpath("lightning.json"), "w") as fid:
        fid.write(job.config)

    # Copy worker_dir to node
    for node in nodes:
        subprocess.run(["-r", worker_dir, node + ":" + worker_dir], "scp")

    # Launch the worker
    args = [
        f"--np={np:d}",
        f"--ppn={job.gpus_per_node:d}",
        f"--hostfile={worker_dir.joinpath('hostfile')}",
        "--cpu-bind=numa",
        "submit/set_node_rank",
        sys.executable,
        "train.py",
        "fit",
        f"--config={worker_dir.joinpath('lightning.json')}",
    ]
    logging.info("mpiexec: %s", args)
    return subprocess.Popen(args, executable=which("mpiexec"))


def init_slurm(sweep_path: str, num_workers):
    workers = []
    ntasks = int(os.environ["SLURM_NTASKS"])
    assert ntasks >= num_workers
    logging.info("starting %d workers", num_workers)
    for id in range(num_workers):

        def worker_launch(job: JobConfig, id=id, sweep_path=sweep_path):
            return srun(job, str(sweep_path), id)

        wkr = Worker(id, worker_launch)
        workers.append(wkr)

    return workers


def init_pbs(num_workers: int):
    workers = []
    with open(os.environ["PBS_NODEFILE"], "r") as fid:
        for node in fid:
            node = node.strip()
            logging.info("starting worker on %s", node)
            wkr = Worker(node, lambda job: qsub(job.config, [node]))
            workers.append(wkr)

    assert len(workers) == num_workers

    return workers


def scheduler(sweep: str, num_workers: int):
    logging.info("populating queue with jobs from %s", sweep)
    queue = Queue(maxsize=0)
    with open(sweep, "r") as fid:
        for lineno, config in enumerate(fid):
            queue.put(JobConfig(lineno + 1, config.strip()))

    # Init workers
    if which("srun") is not None:
        workers = init_slurm(sweep, num_workers)
    elif which("qsub") is not None:
        workers = init_pbs(num_workers)
    else:
        raise RuntimeError("Missing job scheduler")

    # Launch tasks
    slots: dict[Worker, Optional[subprocess.Popen]] = {wrk: None for wrk in workers}
    while queue.not_empty:
        for worker in slots.keys():
            job = slots[worker]
            if job is None:
                job = queue.get()
                logging.info("launching job %s on worker %s", job.lineno, worker.name)
                print(worker.launch)
                slots[worker] = worker.launch(job)
                queue.task_done()

            elif job.poll() is not None:
                if job.returncode == 0:
                    slots[worker] = None
                else:
                    raise RuntimeError(
                        f"Job on worker {worker.name} failed with code {job.returncode}"
                    )
            else:
                logging.info("Worker %s is busy", worker.name)

        sleep(5)

    logging.info("queue complete, shutting down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Subcommand for launching a training job
    subparsers = parser.add_subparsers(dest="subparser")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("sweep", type=str)
    launch_parser.add_argument("id", type=int)
    launch_parser.set_defaults(func=lambda args: launch(args.sweep, args.id))

    sweep_parser = subparsers.add_parser("scheduler")
    sweep_parser.add_argument("sweep", type=str)
    sweep_parser.add_argument("-n", "--num-workers", type=int)
    sweep_parser.set_defaults(func=lambda args: scheduler(args.sweep, args.num_workers))

    args = parser.parse_args()
    args.func(args)
