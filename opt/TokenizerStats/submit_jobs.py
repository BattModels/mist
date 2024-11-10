#!/usr/bin/env python
import time
import logging
import subprocess
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Any
from random import randint
from dataclasses import dataclass, field
from collections import defaultdict
from networkx import DiGraph, topological_sort

logging.basicConfig(level=logging.INFO)

STATS_DIR = Path(__file__).joinpath("..", "stats").resolve()

LOG_FILE = Path(__file__).parent.joinpath("logs", "slurm-%x-%j.log")
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)


@dataclass
class Process:
    cmd: list[str]
    inputs: list[str] = field(default_factory=list)
    output: Optional[Path] = None
    slurm: Optional[dict] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.inputs, list):
            self.inputs = [self.inputs]
        self.inputs = [Path(x) for x in self.inputs]
        self.output = Path(self.output) if self.output is not None else None
        self.slurm = self.slurm or {}

    def launch(self, deps: list[int], dry_run: bool = False) -> int:
        """Launch a slurm job, return the job id"""
        slurm_args = [f"--{k}={v}" for k, v in self.slurm.items()]
        slurm_args.extend(["--output", str(LOG_FILE)])
        for dep in deps:
            assert isinstance(dep, int), "expected int, got %s: %s" % (type(dep), dep)
            slurm_args.extend(["--dependency", f"afterok:{dep}"])

        args = [str(x) for x in self.cmd]
        cmd = ["sbatch", *slurm_args, *args]
        logging.info("launching %s", cmd)

        if not dry_run:
            p = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )
            print(p.stdout)
            print(p.stderr)
            assert p.returncode == 0, f"Non-zero return code: {p.returncode}"
            return int(p.stdout.split(" ")[-1])
        return randint(1, 967_296)


class Workflow:
    def __init__(self):
        self.work = DiGraph()
        self.processes: list[Process] = []

    def add_process(self, *args, **kwargs):
        print(args, kwargs)
        if isinstance(args[0], Process):
            return self._add_process(args[0])
        return self._add_process(Process(*args, **kwargs))

    def _add_process(self, process: Process):
        self.processes.append(process)
        process_id = len(self.processes)
        self.work.add_nodes_from([Path(x) for x in process.inputs])

        if process.output is not None:
            self.work.add_node(Path(process.output))

        for x in process.inputs:
            self.work.add_edge(Path(x), Path(process.output), process_id=process_id)

        return process

    def check(self):
        valid = True

        # Make sure only one process writes to each output
        outputs = defaultdict(list)
        for process in self.processes:
            if process.output is not None:
                outputs[process.output].append(process)

        for output, processes in outputs.items():
            if len(processes) > 1:
                logging.error(f"Multiple processes write to {output}: {processes}")
                valid = False

        # Make sure all inputs exist or are created by a process
        for process in self.processes:
            for input in process.inputs:
                if input in outputs.keys():
                    continue
                elif not input.exists():
                    logging.error(f"Input {input} does not exist")
                    valid = False

        return valid

    def _get_deps(self, process: Process, jobs: dict[Path, int]) -> list[int]:
        deps = []
        for input in process.inputs:
            if input.exists():
                continue
            job_id = jobs[input]
            deps.append(job_id)

        return deps

    def active_jobs(self, file: Path) -> Optional[int]:
        job_file = file.with_suffix(file.suffix + ".slurm")
        if not job_file.exists():
            return None

        # Get the job id
        job_id = job_file.read_text()
        if not job_id:
            job_file.unlink(missing_ok=True)
            return None
        job_id = int(job_id)

        # Get the job state
        p = subprocess.run(
            ["scontrol", "show", "--json", "job", str(job_id)], capture_output=True
        )
        job_info = json.loads(p.stdout)
        try:
            state = job_info["jobs"][0]["job_state"][0]
            if state in ["RUNNING", "PENDING"]:
                return job_id
        except KeyError:
            pass

        except IndexError:
            pass

        # Job is no longer active
        job_file.unlink()
        return None

    def run(self, dry_run=False, rate_limit=100):
        outputs = {
            process.output: process
            for process in self.processes
            if process.output is not None
        }
        jobs: dict[Path, int] = {}
        assert all(isinstance(x, int) for x in jobs.values())
        last_launch = time.time()
        jobs_launched = 0
        tasks_launched = defaultdict(int)
        for file in topological_sort(self.work):
            # Skip input files
            if file not in outputs:
                continue

            if file.exists():
                continue

            if job_id := self.active_jobs(file):
                logging.info("found active job for %s", file)
                jobs[file] = job_id
                continue

            # Get the process that writes to this file
            process = outputs[file]
            deps = self._get_deps(process, jobs)

            # Launch the process
            job_id = process.launch(deps, dry_run)
            assert job_id is not None and isinstance(job_id, int)
            jobs[file] = job_id
            jobs_launched += 1
            tasks_launched[process.meta.get("task", "misc")] += 1

            # Add a marker file with the job id
            if not dry_run:
                file.parent.mkdir(exist_ok=True, parents=True)
                file.with_suffix(file.suffix + ".slurm").write_text(str(job_id))

            # Sleep to avoid hitting the rate limit
            sleep_time = max(1 / rate_limit - (time.time() - last_launch), 0)
            logging.debug(f"ratelimiter: sleeping for {sleep_time} seconds")
            if not dry_run:
                time.sleep(sleep_time)
            last_launch = time.time()

        print(f"Launched {jobs_launched} jobs")
        for task, count in tasks_launched.items():
            print(f"  {task}: {count}")

    def show(self):
        print("Total Processes:", len(self.processes))
        print("Files:", len(self.work))
        print("Valid:", self.check())


MOLNET_DATASETS = [
    "qm8",
    "qm9",
    "esol",
    "freesolv",
    "lipo",
    "muv",
    "hiv",
    "bace",
    "bbbp",
    "tox21",
    "toxcast",
    "sider",
    "clintox",
]

KNOWN_TOKENIZERS = json.loads(Path("tokenizers.json").read_text())

TOKENIZERS = [x["name_or_path"] for x in KNOWN_TOKENIZERS if x["encoding"] == "smile"]

REF_INFO_LOSS = [
    "character",
    "smirk",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-7b",
    "Xenova/gpt-4o",
]


def sbatch(args: list, output: Optional[str] = None, test=True) -> Optional[int]:
    args = [str(x) for x in args]
    if output is not None and Path(output).exists():
        logging.info("skipping job for %s", output)
        return None
    logging.info("submit sbatch: %s => %s", args, output)
    if test:
        return randint(0, 100)

    out = subprocess.run(
        ["/usr/bin/sbatch", *args],
        text=True,
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {out.stderr}")

    return int(out.stdout.split(" ")[-1])


def sub_realspace(tok):
    tok_name = Path(tok).name if Path(tok).is_dir() else tok
    out = STATS_DIR.joinpath(tok_name, "realspace_v4_dev.bson")
    args = [
        "--cpus-per-task=1",
        "--time=1-0:0:0",
        "--ntasks=32",  # Needs to divide datasets evenly (big perf. hit otherwise)
        "submit_tok_stats.sh",
        "usage",
        "--splits=all",
        REALSPACE,
        tok,
    ]
    args = special_case(tok_name, args)

    # id = sbatch(args, out)
    id = None
    return tok_name, id, out


def special_case(tok_name, args):
    if tok_name == "SmilesPE/SPE_ChEMBL":
        args.insert(0, "--mem-per-cpu=4G")
        args.insert(0, "--partition=venkvis-largemem")
    return args


for tok in TOKENIZERS:
    # Submit realspace_v4_dev job
    tok_name, id, realspace_file = sub_realspace(tok)
    ngram_name = realspace_file.with_suffix("").name

    # Compute realspace model loss
    for ds in MOLNET_DATASETS:
        args = [
            "--ntasks=1",
            "--time=4:0:0",
            "--cpus-per-task=1",
            "submit_tok_stats.sh",
            "loss",
            realspace_file,
            ds,
        ]
        if id:
            args.insert(0, f"-d=afterok:{id}")
        args = special_case(tok_name, args)
        if realspace_file.exists():
            sbatch(args, STATS_DIR.joinpath(tok, ds, f"{ngram_name}_model_loss.bson"))

        # Process molnets datsets too
        args = ["--ntasks=1", "--time=1:0:0", "submit_tok_stats.sh", "usage", ds, tok]
        args = special_case(tok_name, args)
        sbatch(args, STATS_DIR.joinpath(tok_name, f"{ds}.bson"))

        # Compute Info loss
        for ref in REF_INFO_LOSS:
            ref_file = STATS_DIR.joinpath(ref, "realspace_v4_dev.bson")
            if not ref_file.exists():
                continue

            args = [
                "--ntasks=48",
                "--cpus-per-task=1",
                "--time=1-0:0:0",
                "submit_tok_stats.sh",
                "distortion",
                f"--reference={ref_file}",
                ds,
                tok,
            ]
            args = special_case(tok_name, args)
            if tok != ref:
                ref_name = ref.replace("/", "--")
                sbatch(
                    args, STATS_DIR.joinpath(tok_name, ds, f"{ref_name}_info_loss.bson")
                )
