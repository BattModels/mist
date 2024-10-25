#!/usr/bin/env python3
import wandb
import argparse
import json
import linecache
import logging
import os
import subprocess
import sys
import sqlite3
from rich.console import Console
from rich.table import Table
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


def launch(args):
    # Load config
    env = os.environ.copy()
    if config_env := args.env:
        config = json.loads(env.pop(config_env))
    else:
        # Load config from json file
        sweep_path = Path(args.sweep_path)
        idx = args.idx
        logging.info("Starting job %d from %s", idx, sweep_path)
        config = json.loads(linecache.getline(str(sweep_path), idx))

    # Patch up slurm environment for lightning
    if ntasks := env.get("SLURM_STEP_NUM_TASKS"):
        env["SLURM_NTASKS"] = ntasks
        env["SLURM_NTASKS_PER_NODE"] = env["SLURM_STEP_TASKS_PER_NODE"]
        env["SLURM_NNODES"] = env["SLURM_STEP_NUM_NODES"]

    # Run the job
    env["PL_CONFIG"] = json.dumps({"fit": config["train"]})
    env["JOB_CONFIG"] = json.dumps(config)
    logging.info("Starting job: %s", config)
    subprocess.run(
        [
            "submit/set_node_rank",
            sys.executable,
            "train.py",
        ],
        check=True,
        env=env,
    )


@dataclass
class JobConfig:
    id: Any
    config: dict

    @property
    def gpus_per_node(self):
        return self.config["gpus_per_node"]

    @property
    def nodes(self):
        return self.config["nodes"]


@dataclass
class Worker:
    name: Any
    launch: Callable[[JobConfig], subprocess.Popen]

    def __hash__(self):
        return hash(self.name)


class SweepDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True, parents=True)
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            "\n".join(
                [
                    "CREATE TABLE IF NOT EXISTS jobs (",
                    "id STRING PRIMARY KEY,",
                    "config TEXT NOT NULL,",
                    "status STRING NOT NULL",
                    ");",
                ]
            )
        )
        con.commit()

    def _connect(self, **kwargs):
        return sqlite3.connect(self.path, **kwargs)

    def __iter__(self):
        con = self._connect()
        cur = con.cursor()
        res = cur.execute("SELECT id, status, config FROM jobs")
        for row in res.fetchall():
            yield {"id": row[0], "status": row[1], "config": json.loads(row[2])}
        con.close()

    def status(self):
        table = Table("id", "status", "ckpt")
        for job in self:
            ckpt = job["config"]["train"].get("ckpt_path", None)
            table.add_row(job["id"], job["status"], ckpt)

        console = Console()
        console.print(table)

    def num_active_jobs(self):
        con = self._connect()
        cur = con.cursor()
        res = cur.execute(
            "SELECT COUNT(*) FROM jobs WHERE status in ('queued', 'running')"
        )
        num = res.fetchone()[0]
        con.close()
        return num

    def queued_jobs(self):
        con = self._connect()
        cur = con.cursor()
        res = cur.execute("SELECT id, config FROM jobs WHERE status = 'queued'")
        for row in res.fetchall():
            yield JobConfig(*row)

    def get_job(self, id: Optional[str] = None) -> JobConfig:
        con = self._connect()
        cur = con.cursor()
        if id is None:
            res = cur.execute(
                "SELECT id, config FROM jobs WHERE status = 'queued' LIMIT 1"
            )
        else:
            res = cur.execute("SELECT id, config FROM jobs WHERE id = ?", (id,))
        id, config = res.fetchone()
        con.close()
        return JobConfig(id, json.loads(config))

    def insert_jobs(self, jobs: list[dict]):
        con = self._connect()
        cur = con.cursor()
        rows = ((str(uuid4()), json.dumps(job), "queued") for job in jobs)
        cur.executemany("INSERT INTO jobs VALUES (?, ?, ?)", rows)
        con.commit()
        con.close()

    def insert_job(self, job: dict, id: Optional[str] = None, status: str = "queued"):
        id = id or str(uuid4())
        con = self._connect()
        cur = con.cursor()
        cur.execute("INSERT INTO jobs VALUES (?, ?, ?)", (id, json.dumps(job), status))
        con.commit()
        con.close()

    def job_status(self, job_id: str):
        con = self._connect()
        cur = con.cursor()
        res = cur.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        status = res.fetchone()[0]
        con.close
        return status

    def update_job_status(self, job_id: str, status: str):
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "UPDATE OR ABORT jobs SET status = ? WHERE id = ?", (status, job_id)
        )
        con.commit()
        con.close()
        return status

    def set_ckpt_path(self, job_id: str, path: Path):
        job = self.get_job(job_id)
        job.config["train"]["ckpt_path"] = str(path)
        logging.info("Setting checkpoint path for %s to %s", job_id, path)
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "UPDATE OR ABORT jobs SET config = ? where id = ?",
            (json.dumps(job.config), job.id),
        )
        con.commit()
        con.close()
        return self.get_job(job_id)


def setup_wandb(job: JobConfig, env: dict):
    env["WANDB_RUN_ID"] = str(job.id)
    env["WANDB_RESUME"] = "allow"
    return env


def srun(job: JobConfig, worker_id: int):
    logging.info("Running job %s: %s", job.id, job.config)
    container = job.config.get(
        "container", "/lustre/fs0/shared/sqsh-files/0535844560745234+mist+latest.sqsh"
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
        f"--output=slurm-%j.{job.id}.out",
        "--open-mode=append",
        "--export=ALL",
        "/mist/.venv/bin/python3",
        __file__,
        "launch",
        "--env=JOB_CONFIG",
    ]
    env = os.environ.copy()
    env["JOB_CONFIG"] = json.dumps(job.config)
    env = setup_wandb(job, env)
    logging.info("srun: %s", srun_args)
    return subprocess.Popen(srun_args, executable=which("srun"), env=env)


def qsub(job: JobConfig, nodes: list[str]):
    gpus_per_node = job.config.get("gpus_per_node", 4)
    np = len(nodes) * gpus_per_node
    assert (
        len(nodes) == job.nodes
    ), f"Expected node counts to match got {len(nodes)} and {job.nodes}"

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    worker_dir = tmpdir.joinpath(str(uuid4()))
    worker_dir.mkdir()
    with open(worker_dir.joinpath("hostfile"), "w") as fid:
        for node in nodes:
            fid.write(node + "\n")

    # Launch the worker
    args = [
        f"--np={np:d}",
        f"--ppn={gpus_per_node:d}",
        f"--hostfile={worker_dir.joinpath('hostfile')}",
        "--cpu-bind=numa",
        __file__,
        "launch",
        "--env=JOB_CONFIG",
    ]
    env = os.environ.copy()
    env["JOB_CONFIG"] = json.dumps(job.config)
    env = setup_wandb(job, env)
    logging.info("mpiexec: %s", args)
    return subprocess.Popen(
        args,
        executable=which("mpiexec"),
        env=env,
    )


def init_slurm(num_workers: int):
    workers = []
    ntasks = int(os.environ["SLURM_NTASKS"])
    assert ntasks >= num_workers
    logging.info("starting %d workers", num_workers)
    for id in range(num_workers):

        def worker_launch(job: JobConfig, id=id):
            return srun(job, id)

        wkr = Worker(id, worker_launch)
        workers.append(wkr)

    return workers


def init_pbs(num_workers: int):
    workers = []
    with open(os.environ["PBS_NODEFILE"], "r") as fid:
        for node in fid:
            node = node.strip()
            logging.info("starting worker on %s", node)

            def worker_launch(job: JobConfig, nodes=[node]):
                return qsub(job, nodes)

            workers.append(Worker(node, worker_launch))

    assert len(workers) == num_workers

    return workers


def scheduler(sweep: str, num_workers: int, run_dir: str = "mist"):
    logging.info("populating queue with jobs from %s", sweep)
    queue = SweepDB(sweep)

    # Mark all unfinished jobs as queued
    for job in queue:
        logging.debug("reevaluating state of job %s: %s", job["id"], job["config"])
        if job["status"] in ["finished", "failed"]:
            continue

        if job["status"] not in ["queued", "failed"]:
            logging.info("marking job %s as %s -> queued", job["id"], job["status"])
            queue.update_job_status(job["id"], "queued")
            ckpt_path = Path(run_dir, job["id"], "checkpoints", "last.ckpt")
            if ckpt_path.exists():
                logging.debug("setting ckpt path for %s to %s", job["id"], ckpt_path)
                queue.set_ckpt_path(job["id"], ckpt_path)

    # Init workers
    if which("srun") is not None:
        workers = init_slurm(num_workers)
    elif which("qsub") is not None:
        workers = init_pbs(num_workers)
    else:
        raise RuntimeError("Missing job scheduler")

    # Launch tasks
    slots: dict[Worker, Optional[Any]] = {wrk: None for wrk in workers}
    while queue.num_active_jobs() > 0:
        for worker in slots.keys():
            item = slots[worker]

            # Add a new job if there is none
            if item is None:
                job = queue.get_job()
                logging.info("launching job %s on worker %s", job.id, worker.name)
                slots[worker] = (job, worker.launch(job))
                queue.update_job_status(job.id, "running")
                continue

            # Check if the process is finished
            job, proc = item
            if proc.poll() is not None:
                if proc.returncode == 0:
                    slots[worker] = None
                    queue.update_job_status(job.id, "finished")
                else:
                    queue.update_job_status(job.id, "failed-exit-code")
                    raise RuntimeError(
                        f"Job on worker {worker.name} failed with code {proc.returncode}"
                    )
            else:
                logging.info("Worker %s is busy", worker.name)

        sleep(5)

    logging.info("queue complete, shutting down")


def create_sweep_job(
    base_config: dict, jobs: list[dict], num_workers: Optional[int] = None
):
    """
        config = create_sweep_job(base_config, jobs, num_workers = None)

    Returns a job config that when rendered will launch a multi-node sweep of the given jobs.

    base_config: The base config to use for the sweep. All jobs must have the same number of nodes and gpus per node.
    jobs: A list of job configs to run in the sweep.
    num_workers: The number of workers (nodes) to simultaneously run jobs

    """
    # Check that all jobs have the same number of nodes and gpus per node
    gpus_per_node = base_config.get("gpus_per_node", None)
    nodes = base_config.get("nodes", None)
    for job in jobs:
        if gpus_per_node is not None:
            assert job["gpus_per_node"] == gpus_per_node
        else:
            assert job.get("gpus_per_node", None) is None
        if nodes is not None:
            assert job["nodes"] == nodes
        else:
            assert job.get("nodes", None) is None

    # Create the sweep queue database
    queue = SweepDB(Path(".cache", "sweep", f"{uuid4()}.sqlite3"))
    queue.insert_jobs(jobs)

    num_workers = num_workers or len(jobs)
    base_config["job_queue"] = {"tasks": num_workers, "queue": queue.path}
    return base_config


def remap_wandb_state(state: str) -> str:
    if state in ["crashed", "failed"]:
        return "queued"
    elif state == "running":
        return "running"
    elif state == "finished":
        return "finished"
    else:
        raise RuntimeError(f"Unknown state {state}")


def create_sweep_from_tags(
    tags: list[str],
    project: str = "incite-mist/mist",
    run_dir: str = "mist",
):
    api = wandb.Api()
    run_dir = Path(run_dir).resolve()
    queue = SweepDB(Path(".cache", "sweep", f"{uuid4()}.sqlite3"))
    logging.info("Creating sweep database at %s", queue.path)
    for run in api.runs(project, filters={"tags": {"$in": tags}}):
        status = remap_wandb_state(run.state)
        config = run.config["job_config"]
        if not config:
            job_config = Path(run_dir, run.id, "job_config.json")
            if job_config.is_file():
                config = json.loads(job_config.read_text())
                print(config)
                logging.warn("Using job_config from %s for %s", job_config, run.id)

        if not config:
            logging.error("Missing job_config for run %s, skipping", run.id)
            continue

        queue.insert_job({"id": run.id, "config": config, "status": status})

    # Display queue
    logging.info("Created sweep database: %s", queue.path)
    queue.status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Subcommand for launching a training job
    subparsers = parser.add_subparsers(dest="subparser")
    launch_parser = subparsers.add_parser("launch")
    launch_type = launch_parser.add_mutually_exclusive_group(required=True)
    launch_type.add_argument("--env", type=str)
    launch_type.add_argument("--sweep", type=str)
    launch_parser.add_argument("--id", type=int)
    launch_parser.set_defaults(func=lambda args: launch(args))

    sweep_parser = subparsers.add_parser("scheduler")
    sweep_parser.add_argument("sweep", type=str)
    sweep_parser.add_argument("-n", "--num-workers", type=int)
    sweep_parser.set_defaults(func=lambda args: scheduler(args.sweep, args.num_workers))

    queue_status_parser = subparsers.add_parser("queue-status")
    queue_status_parser.add_argument("sweep", type=str)
    queue_status_parser.set_defaults(func=lambda args: SweepDB(args.sweep).status())

    wandb_parser = subparsers.add_parser("wandb")
    wandb_parser.add_argument("tags", type=str, nargs="+")
    wandb_parser.add_argument("--project", type=str, default="incite-mist/mist")
    wandb_parser.add_argument("--run-dir", type=str, default="./mist")
    wandb_parser.set_defaults(
        func=lambda args: create_sweep_from_tags(
            args.tags, project=args.project, run_dir=args.run_dir
        )
    )

    args = parser.parse_args()
    args.func(args)
