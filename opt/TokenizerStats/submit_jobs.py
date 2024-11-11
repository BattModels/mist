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

REF_INFO_LOSS = [
    "character",
    "smirk",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-7b",
    "Xenova/gpt-4o",
]


def usage(dataset, tokenizer, ds_name=None, slurm={}, encoding="smiles"):
    ds_name = ds_name or dataset
    output = STATS_DIR.joinpath(tokenizer, ds_name, "usage.bson")
    if tokenizer == "SmilesPE/SPE_ChEMBL":
        slurm["--mem-per-cpu"] = "8G"
        slurm["--partition"] = "venkvis-largemem"

    return Process(
        [
            "submit_tok_stats.sh",
            "usage",
            "--splits=all",
            "--encoding",
            encoding,
            f"--output={output}",
            str(dataset),
            tokenizer,
        ],
        output=output,
        slurm=slurm,
        meta={"dataset": dataset, "tokenizer": tokenizer, "task": "usage"},
    )


def ngram_loss(dataset, tokenizer, slurm={}, encoding="smiles"):
    input = STATS_DIR.joinpath(tokenizer, "realspace", "usage.bson")
    output = STATS_DIR.joinpath(tokenizer, dataset, "model_loss.bson")
    if tokenizer == "SmilesPE/SPE_ChEMBL":
        slurm["--mem-per-cpu"] = "8G"
        slurm["--partition"] = "venkvis-largemem"

    return Process(
        [
            "submit_tok_stats.sh",
            "loss",
            "--encoding",
            encoding,
            "--output",
            output,
            "--model",
            input,
            dataset,
            tokenizer,
        ],
        inputs=input,
        output=output,
        slurm=slurm,
        meta={"dataset": dataset, "tokenizer": tokenizer, "task": "model_loss"},
    )


def ngram_info_loss(dataset, tokenizer, ref, slurm={}, encoding="smiles"):
    ref_name = ref.replace("/", "--")
    ref_usage = STATS_DIR.joinpath(ref, "realspace", "usage.bson")
    output = STATS_DIR.joinpath(tokenizer, dataset, f"{ref_name}_info_loss.bson")
    if tokenizer == "SmilesPE/SPE_ChEMBL":
        slurm["--mem-per-cpu"] = "8G"
        slurm["--partition"] = "venkvis-largemem"

    return Process(
        [
            "submit_tok_stats.sh",
            "distortion",
            "--encoding",
            encoding,
            "--output",
            output,
            "--reference",
            ref_usage,
            dataset,
            tokenizer,
        ],
        inputs=ref_usage,
        output=output,
        slurm=slurm,
        meta={"dataset": dataset, "tokenizer": tokenizer, "task": "info_loss"},
    )


def set_logging_level(verbosity):
    # Map verbosity count to logging levels
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, len(levels) - 1)]  # Cap to the highest level
    logging.basicConfig(level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--realspace", type=str, default="/nfs/turbo/coe-venkvis/mist/realspace_v4_dev2"
    )
    args = parser.parse_args()
    set_logging_level(args.verbose)

    tokenizers = json.loads(Path("tokenizers.json").read_text())
    wk = Workflow()
    for tok in tokenizers:
        tok_name = tok["name_or_path"]
        # Tabulate OOVs
        output = STATS_DIR.joinpath(tok_name, "oov.json")
        p = wk.add_process(
            ["submit_oov.sh", "--output", output, tok_name], output=output
        )

        # Tokenize RealSpace
        wk.add_process(
            usage(
                args.realspace,
                tok_name,
                ds_name="realspace",
                encoding=tok["encoding"],
                slurm={"ntasks": 32, "time": "1-0:0:0"},
            )
        )
        src = STATS_DIR.joinpath(tok_name, "realspace_v4_dev.bson")
        if src.exists():
            STATS_DIR.joinpath(tok_name, "realspace").mkdir(exist_ok=True)
            shutil.move(src, STATS_DIR.joinpath(tok_name, "realspace", "usage.bson"))

        # Tokenize MoleculeNet
        for ds in MOLNET_DATASETS:
            wk.add_process(
                usage(
                    ds,
                    tok_name,
                    encoding=tok["encoding"],
                    slurm={"ntasks": 1, "time": "1:0:0"},
                )
            )
            wk.add_process(
                ngram_loss(
                    ds,
                    tok_name,
                    encoding=tok["encoding"],
                    slurm={"ntasks": 1, "time": "2:0:0"},
                )
            )
            src = STATS_DIR.joinpath(tok_name, f"{ds}.bson")
            if src.exists():
                shutil.move(src, STATS_DIR.joinpath(tok_name, ds, "usage.bson"))

            for ref in REF_INFO_LOSS:
                wk.add_process(
                    ngram_info_loss(
                        ds,
                        tok_name,
                        ref,
                        encoding=tok["encoding"],
                        slurm={"ntasks": 24, "time": "4:0:0"},
                    )
                )

        wk.add_process(
            usage(
                "tmQM",
                tok_name,
                encoding=tok["encoding"],
                slurm={"ntasks": 4, "time": "1:0:0"},
            )
        )
        wk.add_process(
            ngram_loss(
                "tmQM",
                tok_name,
                encoding=tok["encoding"],
                slurm={"ntasks": 1, "time": "2:0:0"},
            )
        )
        for ref in REF_INFO_LOSS:
            wk.add_process(
                ngram_info_loss(
                    "tmQM",
                    tok_name,
                    ref,
                    encoding=tok["encoding"],
                    slurm={"ntasks": 48, "time": "4:0:0"},
                )
            )

    wk.show()
    wk.run(dry_run=args.dry_run)
