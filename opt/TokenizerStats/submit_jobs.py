#!/usr/bin/env -S uv run python
import re
import argparse
import json
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Any, Optional

from networkx import DiGraph, topological_sort

logging.basicConfig(level=logging.INFO)

STATS_DIR = Path(__file__).joinpath("..", "stats-encoding").resolve()

LOG_FILE = Path(__file__).parent.joinpath("logs", "slurm-%x-%j.log")
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)


@dataclass
class Process:
    cmd: list[str]
    inputs: list[Path] = field(default_factory=list)
    output: Optional[Path] = None
    slurm: Optional[dict] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.inputs, list):
            self.inputs = [self.inputs]
        self.inputs = [Path(x) for x in self.inputs]
        self.output = Path(self.output) if self.output is not None else None
        if self.output is not None:
            self.output = self.output.resolve()
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

            # Add a marker file with the job id
            job_id = int(p.stdout.split(" ")[-1])
            if output := self.output:
                output.parent.mkdir(exist_ok=True, parents=True)
                output.with_suffix(output.suffix + ".slurm").write_text(str(job_id))

            return job_id

        return randint(1, 967_296)

    def active_job(self) -> int | None:
        if self.output is None:
            return None

        job_file = self.output.with_suffix(self.output.suffix + ".slurm")
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


class Workflow:
    def __init__(self):
        self.work = DiGraph()
        self.processes: list[Process] = []

    def add_process(self, *args, **kwargs):
        if isinstance(args[0], Process):
            return self._add_process(args[0])
        return self._add_process(Process(*args, **kwargs))

    def _add_process(self, process: Process):
        self.processes.append(process)
        process_id = len(self.processes)
        self.work.add_nodes_from([Path(x) for x in process.inputs])

        if process.output is not None:
            output = Path(process.output)
            if self.work.has_node(output):
                if pd := list(self.work.predecessors(output)):
                    raise RuntimeError(
                        f"Duplicate output {output}:\n{process}\n\n {pd}"
                    )

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

    def launch(self, process: Process, deps, dry_run: bool = False):
        # Launch the process
        job_id = process.launch(deps, dry_run)
        assert job_id is not None and isinstance(job_id, int)

        return job_id

    def run(
        self,
        dry_run=False,
        rate_limit=100,
        preflight=list[Process],
        postflight=list[Process],
    ):
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

        preflight_ids = []
        for process in preflight:
            preflight_ids.append(process.launch([], dry_run))

        for file in topological_sort(self.work):
            # Skip input files
            if file not in outputs:
                continue
            process = outputs[file]

            # Check for an active job and clean up marker files
            if job_id := process.active_job():
                logging.info("found active job for %s", file)
                jobs[file] = job_id
                continue

            if file.exists():
                continue

            # Get the dependencies for this process
            deps = self._get_deps(process, jobs)
            deps.extend(preflight_ids)
            deps = list(set(deps))

            # Launch the process
            jobs[file] = self.launch(process, deps, dry_run)
            jobs_launched += 1
            tasks_launched[process.meta.get("task", "misc")] += 1

            # Sleep to avoid hitting the rate limit
            sleep_time = max(1 / rate_limit - (time.time() - last_launch), 0)
            logging.debug(f"ratelimiter: sleeping for {sleep_time} seconds")
            if not dry_run:
                time.sleep(sleep_time)
            last_launch = time.time()

        # Launch postflight jobs
        for process in postflight:
            if process.active_job() is not None:
                logging.info("found active job for %s", process.output)
                continue

            process.launch(jobs.values(), dry_run)

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
    # "meta-llama/Meta-Llama-3.1-8B",
    # "google/gemma-7b",
    # "Xenova/gpt-4o",
]

LARGE_MEM_TOKENIZERS = [
    "mikemayuare/SMILYAPE",
    "mikemayuare/SELFYAPE",
    "mikemayuare/SMILYBPE",
    "mikemayuare/SELFYBPE",
    "SmilesPE/SPE_ChEMBL",
]


def large_mem(tokenizer: str, slurm: dict) -> dict:
    if tokenizer in LARGE_MEM_TOKENIZERS:
        slurm["mem-per-cpu"] = "32G"
        slurm["partition"] = "venkvis-largemem,venkvis-cpu"
    return slurm


def find_incomplete_ranks(outputs: list[Path], n_jobs: int) -> set[int]:
    re_complete = re.compile(r"rank_(\d+)\.jld2$")
    complete = set()
    for output in outputs:
        if m := re_complete.search(output.name):
            complete.add(int(m.group(1)))
    return set(range(n_jobs)) - complete


def usage(dataset, tokenizer, ds_name=None, slurm=None, encoding="smiles", mode="mpi"):
    ds_name = ds_name or str(dataset)
    output = STATS_DIR.joinpath(tokenizer, ds_name, f"usage_{encoding}.jld2")
    slurm = slurm or {}
    slurm["job-name"] = slurm.get("job-name", f"usage-{ds_name}")

    slurm.setdefault("job-name", f"usage-{ds_name}")
    slurm.setdefault("ntasks", 4)
    slurm.setdefault("time", "1-0:0:0")
    slurm.setdefault("mem-per-cpu", "1800M")
    slurm.setdefault("partition", "venkvis-cpu,venkvis-largemem")

    world_size = int(slurm["ntasks"])
    if mode == "batch":
        output = Path(output.parent, ".unmerged", f"usage_{encoding}", output.name)
        slurm["array"] = f"0-{slurm['ntasks'] - 1}"

        # Check if all array outputs are present
        n_complete = len(list(output.parent.glob("*.jld2")))
        logging.debug("found %d array output files for %s", n_complete, output.parent)
        witness = output.parent.with_suffix(".witness")
        if n_complete == slurm["ntasks"]:
            witness.touch()
        else:
            witness.unlink(missing_ok=True)
            incomplete_ranks = find_incomplete_ranks(
                list(output.parent.glob("*.jld2")), slurm["ntasks"]
            )
            if len(incomplete_ranks) < slurm["ntasks"]:
                slurm["array"] = ",".join(str(x) for x in incomplete_ranks)

        slurm["ntasks"] = 1

    return Process(
        [
            "submit_tok_stats.sh",
            "usage",
            f"--mode={mode}",
            f"--size={world_size}",
            "--splits=all",
            "--encoding",
            encoding,
            f"--output={output}",
            str(dataset),
            tokenizer,
        ],
        output=output.parent.with_suffix(".witness") if mode == "batch" else output,
        slurm=slurm,
        meta={"dataset": dataset, "tokenizer": tokenizer, "task": "usage"},
    )


def usage_array(
    wk: Workflow, dataset, tokenizer, ds_name=None, slurm=None, encoding="smiles"
):
    p = usage(
        dataset,
        tokenizer,
        ds_name=ds_name,
        slurm=slurm,
        encoding=encoding,
        mode="batch",
    )
    output = p.output.parent.parent.joinpath(f"usage_{encoding}.jld2")
    slurm = {
        "job-name": "merge-usage",
        "partition": "venkvis-cpu,venkvis-largemem",
        "mem-per-cpu": "64G",
        "cpus-per-task": 1,
        "ntasks": 1,
        "time": "2:0:0",
    }
    wk._add_process(p)
    wk.add_process(
        ["submit_tok_stats.sh", "merge", p.output.with_suffix(""), output],
        inputs=[p.output],
        output=output,
        slurm=slurm,
        meta={**p.meta, "task": "merge"},
    )


def ngram_loss(
    dataset, tokenizer, slurm=None, encoding="smiles", ngram="realspace", ds_name=None
):
    input = STATS_DIR.joinpath(tokenizer, ngram, f"usage_{encoding}.jld2")
    ds_name = ds_name or str(dataset)
    outfile = f"model_loss_{ngram}_{encoding}.jld2"
    output = STATS_DIR.joinpath(tokenizer, ds_name, outfile)
    slurm = slurm or {}
    if ngram == "realspace":
        slurm.setdefault("job-name", f"loss-{ds_name}")
    else:
        slurm.setdefault("job-name", f"loss-{ds_name}-{ngram}")

    slurm.setdefault("mem-per-cpu", "1800M")
    slurm.setdefault("partition", "venkvis-cpu,venkvis-largemem")
    slurm.setdefault("ntasks", 4)
    slurm.setdefault("time", "2:0:0")
    slurm = large_mem(tokenizer, slurm)

    return Process(
        [
            "submit_tok_stats.sh",
            "loss",
            "--encoding",
            encoding,
            "--output",
            str(output),
            "--model",
            str(input),
            dataset,
            tokenizer,
        ],
        inputs=[input],
        output=output,
        slurm=slurm,
        meta={"dataset": dataset, "tokenizer": tokenizer, "task": "model_loss"},
    )


def ngram_info_loss(
    dataset, tokenizer, ref, slurm=None, encoding="smiles", ds_name=None
):
    ref_name = ref.replace("/", "--")
    ref_usage = STATS_DIR.joinpath(ref, "realspace", f"usage_{encoding}.jld2")
    ds_name = ds_name or str(dataset)
    output = STATS_DIR.joinpath(
        tokenizer, ds_name, f"{ref_name}_info_loss_{encoding}.jld2"
    )
    slurm = slurm or {}
    slurm.setdefault("job-name", f"dist-{ds_name}")
    slurm.setdefault("mem-per-cpu", "3600M")
    slurm.setdefault("partition", "venkvis-cpu,venkvis-largemem")
    slurm.setdefault("ntasks", 4)
    slurm.setdefault("time", "1-0:0:0")

    slurm = large_mem(tokenizer, slurm)

    return Process(
        [
            "submit_tok_stats.sh",
            "distortion",
            "--encoding",
            encoding,
            "--output",
            str(output),
            "--reference",
            str(ref_usage),
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


def tokenizer_jobs(wk, tok, realspace_path, tmqm_path):
    tok_name = tok["name_or_path"]

    # Tokenize RealSpace
    if tok_name in LARGE_MEM_TOKENIZERS:
        usage_array(
            wk,
            realspace_path,
            tok_name,
            ds_name="realspace",
            encoding=tok["encoding"],
            slurm={"ntasks": 128, "time": "1-0:0:0", "mem-per-cpu": "6G"},
        )
    else:
        wk.add_process(
            usage(
                realspace_path,
                tok_name,
                ds_name="realspace",
                encoding=tok["encoding"],
                slurm={"ntasks": 32, "time": "1-0:0:0"},
            )
        )

    wk.add_process(
        ngram_loss(
            realspace_path,
            tok_name,
            ds_name="realspace",
            encoding=tok["encoding"],
            slurm={"ntasks": 128, "time": "8:0:0"},
        )
    )
    wk.add_process(
        ngram_info_loss(
            realspace_path,
            tok_name,
            ref="character",
            ds_name="realspace",
            encoding=tok["encoding"],
            slurm={"ntasks": 128, "time": "1-0:0:0"},
        )
    )

    # Tokenize MoleculeNet
    for ds in [*MOLNET_DATASETS, "tmqm"]:
        dataset = ds if ds != "tmqm" else tmqm_path
        ds_name = None if ds != "tmqm" else "tmqm"
        wk.add_process(
            usage(
                dataset,
                tok_name,
                ds_name=ds_name,
                encoding=tok["encoding"],
            )
        )
        for ngram in ["realspace", ds]:
            wk.add_process(
                ngram_loss(
                    dataset,
                    tok_name,
                    encoding=tok["encoding"],
                    ngram=ngram,
                    ds_name=ds_name,
                    slurm={"time": "8:0:0" if ds == "tmqm" else "4:0:0"},
                )
            )

        for ref in REF_INFO_LOSS:
            if ds == "tmqm" and tok["encoding"] == "selfies":
                continue  # Majority of selfies fail

            wk.add_process(
                ngram_info_loss(
                    dataset,
                    tok_name,
                    ref,
                    ds_name=ds_name,
                    encoding=tok["encoding"],
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--precompile", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--realspace", type=str, default="/nfs/turbo/coe-venkvis/mist/realspace_v4_dev2"
    )
    parser.add_argument(
        "--tmqm",
        type=str,
        default=Path(__file__).parent.parent.joinpath("tmQM", "data"),
    )
    args = parser.parse_args()
    set_logging_level(args.verbose)

    tokenizers = json.loads(Path("tokenizers.json").read_text())
    wk = Workflow()
    for tok in tokenizers:
        # Tabulate OOVs
        output = STATS_DIR.joinpath(tok["name_or_path"], "oov.json")
        wk.add_process(
            ["submit_oov.sh", "--output", output, tok["name_or_path"]],
            output=output,
            meta={"task": "oov"},
        )

        if tok["name_or_path"] == "character":
            encodings = ["smiles", "selfies", "smiles-canonical", "smiles-kekule"]
        elif tok["encoding"] == "selfies":
            encodings = ["selfies"]
        else:
            encodings = ["smiles", "smiles-canonical", "smiles-kekule"]

        for encoding in encodings:
            tok["encoding"] = encoding
            tokenizer_jobs(wk, tok, args.realspace, args.tmqm)

    wk.show()
    preflight = []
    if args.precompile:
        preflight.append(Process(["submit_precompile.sh"], []))

    postflight = [
        Process(["submit_archive.sh"], output=Path("archive")),
    ]

    wk.run(dry_run=args.dry_run, preflight=preflight, postflight=postflight)
