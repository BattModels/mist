# Configure TLS only at application startup, before HTTP clients are imported.
if __name__ == "__main__":
    import truststore

    truststore.inject_into_ssl()

import argparse
import functools
import gzip
import json
import logging
import multiprocessing
from collections.abc import Callable
from os import environ
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Optional

import accelerate
import torch
from assembly_theory import molecular_assembly
from BRSAScore import SAScorer as BRSAScorer
from datasets import Dataset, load_dataset, load_from_disk
from rdkit import Chem, rdBase
from rdkit.Chem.Descriptors import MolWt
from rdkit.Contrib.SA_Score import sascorer
from sklearn.metrics import roc_auc_score
from smirk import SmirkTokenizerFast
from syba.syba import SybaClassifier
from vendor.scscore.scscore import SCScorer

from electrolyte_fm.data_modules.utils import MolEncoding, encode_molecules
from electrolyte_fm.models.mol_surprise import MolSurpriseFM
from electrolyte_fm.utils.cache import cached_download, extract_file

# Suppress DeprecationWarnings for MorganGenerator
rdBase.DisableLog("rdApp.warning")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_PROCS = int(environ.get("SLURM_CPUS_PER_TASK", "4"))


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target(q, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    q.put(result)
                except Exception as e:  # noqa: BLE001 - forward child failures to the parent
                    q.put(e)

            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=target, args=(q, *args), kwargs=kwargs)
            p.start()
            p.join(seconds)

            if p.is_alive():
                p.terminate()
                p.join()
                logger.warning(
                    f"Function '{func.__name__}' timed out after {seconds} seconds with args={args}, kwargs={kwargs}"
                )
                return None

            try:
                result = q.get(timeout=1)
            except Empty:
                logger.warning(
                    "Worker for %s exited without a result (exit %s)",
                    func.__name__,
                    p.exitcode,
                )
                return None
            finally:
                q.close()
            if isinstance(result, Exception):
                raise result
            return result

        return wrapper

    return decorator


def sascore(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def syba_scorer():
    # Directions for obtaining count files: https://github.com/lich-uct/syba/issues/6
    archive = cached_download(
        "https://anaconda.org/LICH/syba/1.0.1/download/noarch/syba-1.0.1-py_0.tar.bz2",
        Path("syba", "syba-1.0.1-py_0.tar.bz2"),
    )
    count_file = Path(archive).parent.joinpath("syba.csv.gz")
    if not count_file.is_file():
        extract_file(archive, "syba/resources/syba.csv.gz", count_file)

    syba = SybaClassifier()
    with gzip.open(count_file, "rt") as f:
        syba.fitFromCountFile(f)
    return syba


@timeout(30)  # molecular_assembly (v0.2.0) timeout flag doesn't timeout
def molecular_assembly_timeout(smi: str) -> int | None:
    mol = Chem.MolFromSmiles(smi)
    will_error = [
        "CC(C)(C)OC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",  # Panics in isomorphism.rs
    ]
    if smi in will_error:
        return None
    try:
        return molecular_assembly(mol)
    except Exception as e:  # noqa: BLE001 - a failed heuristic is recorded as missing
        # Catch panics
        logger.error("molecular assembly threw an error for %s: %s", smi, e)
        return None


def molecular_weight(smiles: str) -> float:
    """
    Calculate the molecular weight of a molecule given its SMILES string.

    Parameters:
    smiles (str): The SMILES representation of the molecule.

    Returns:
    float: Molecular weight of the molecule, or None if invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolWt(mol)


def map_batchsize_finder(ds, f, batch_size: int = 64, **kwargs):
    while batch_size >= 1:
        try:
            with torch.inference_mode():
                return ds.map(f, batch_size=batch_size, **kwargs)
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size // 2
            torch.cuda.empty_cache()

    raise RuntimeError("Failed to find a batch size that doesn't cause an OOM")


def eval_fm_model(
    metric_name: str, model: MolSurpriseFM, ds: Dataset, target: str | None = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.eval()
    runtime = {}
    auroc_stats = {}
    for encoding in ["smiles", "smiles-kekule", "smiles-canonical"]:
        for per_token in [False, True]:
            name = f"{metric_name}-{encoding}"
            if per_token:
                name += "-per-token"

            start = perf_counter()

            def scorer(x, name=name, per_token=per_token, model=model):
                return {name: model.score(x, per_token=per_token).tolist()}

            ds = map_batchsize_finder(
                ds,
                scorer,
                batched=True,
                batch_size=8,
                input_columns=encoding,
                desc=name,
            )
            runtime[name] = perf_counter() - start
            if target is not None:
                auroc_stats[name] = roc_auc_score(ds[target], ds[name])

    # Flush model from memory
    del model
    torch.cuda.empty_cache()

    return ds, runtime, auroc_stats


def evaluate_dataset(
    metrics: dict[str, Callable | str],
    ds: Dataset,
    target: str | None = None,
    smi_column: str = "smiles",
):
    if smi_column != "smiles":
        ds = ds.rename_column(smi_column, "smiles")
        smi_column = "smiles"

    ds = encode_molecules(
        ds,
        "smiles",
        output_column="smiles-kekule",
        encoding=MolEncoding.KEKULE,
        num_proc=NUM_PROCS,
    )
    ds = encode_molecules(
        ds,
        "smiles",
        output_column="smiles-canonical",
        encoding=MolEncoding.CANONICAL_SMILES,
        num_proc=NUM_PROCS,
    )

    stats = {"auroc": {}}
    stats["time"] = {}
    for name, metric in metrics.items():
        if isinstance(metric, str):
            for suffix, init_model in [
                ("", MolSurpriseFM.from_pretrained),
                ("-untrained", MolSurpriseFM.from_untrained),
            ]:
                ds, metric_runtime, metric_auroc = eval_fm_model(
                    name + suffix,
                    init_model(metric),
                    ds,
                    target,
                )
                stats["time"].update(metric_runtime)
                stats["auroc"].update(metric_auroc)

        else:
            start = perf_counter()

            # Allow for different input encodings
            if "kekule" in name:
                input_columns = "smiles-kekule"
            elif "canonical" in name:
                input_columns = "smiles-canonical"
            else:
                input_columns = "smiles"

            ds = ds.map(
                lambda x, name=name, metric=metric: {name: metric(x)},
                input_columns=input_columns,
                batched=False,
                desc=name,
                num_proc=1 if name == "assembly-index" else NUM_PROCS,
            )
            stats["time"][name] = perf_counter() - start
            if target:
                ds_auroc = ds.filter(
                    lambda x: x is not None, batched=False, input_columns=name
                )
                stats["auroc"][name] = roc_auc_score(ds_auroc[target], ds_auroc[name])

    # Arrow retains a timed-out heuristic as None. A pandas round trip would
    # change it to NaN, which is invalid JSON and breaks the Julia report.
    stats["scores"] = ds.to_list()

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", type=str, action="append", default=None)
    p.add_argument("--skip-metric", type=str, action="append", default=None)
    p.add_argument("--output-format", type=str, default="{dataset}.json")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument(
        "--model", nargs=2, action="append", default=[], metavar=("NAME", "MODEL_DIR")
    )
    p.add_argument("--dataset", choices=["ba-sascorer", "crowdsourced"])
    args = p.parse_args()
    torch.manual_seed(42)

    selected_metrics = set(args.metric or [])

    def need(name):
        return (
            args.prepare
            or (not selected_metrics and not args.model)
            or name in selected_metrics
        )

    syba = syba_scorer() if need("SyBA") else None
    scscore = SCScorer().restore() if need("SCScore") else None
    ba_sascorer = BRSAScorer() if need("BR-SAScore") else None
    smirk = SmirkTokenizerFast()
    metrics = {
        "SAScore": sascore,
        "SyBA": lambda smi: -syba.predict(smi),
        "SCScore": lambda smi: scscore.get_score_from_smi(smi)[1],
        "BR-SAScore": lambda smi: ba_sascorer.calculateScore(smi)[0],
        "MolFormer": "ibm-research/MoLFormer-XL-both-10pct",
        "ChemBERTa": "seyonec/ChemBERTa-zinc-base-v1",
        "smirk": lambda smi: len(smirk(smi)["input_ids"]),
        "smirk-kekule": lambda smi: len(smirk(smi)["input_ids"]),
        "smirk-canonical": lambda smi: len(smirk(smi)["input_ids"]),
        "molecular-weight": molecular_weight,
        "assembly-index": molecular_assembly_timeout,
    }

    for file in Path("models").iterdir() if Path("models").is_dir() else []:
        if file.is_dir():
            metrics[file.name] = str(file)

    if args.model:
        metrics = dict(args.model)

    # Uncomment to just run assembly-index
    if args.metric is not None:
        metrics = {metric: metrics[metric] for metric in args.metric}

    if args.skip_metric is not None:
        for metric in args.skip_metric:
            metrics.pop(metric, None)
    logger.info("Evaluating: %s", ",".join(metrics.keys()))

    sources = {
        # Retained in the pinned scorer fork after removal from upstream main.
        "ba-sascorer": "https://raw.githubusercontent.com/awadell1/BR-SAScore/464a26335b44d93e216d56d239dc3e942d563d46/data/test_set.csv",
        # The same public ACS object served by Figshare file 3917065. Some HPC
        # networks reset connections to Figshare's redirector, but allow S3.
        "crowdsourced": "https://s3-eu-west-1.amazonaws.com/pstorage-acs-6854636/3917065/ci5001778_si_001.txt",
    }
    for dataset, url in sources.items():
        if args.dataset and args.dataset != dataset:
            continue
        path = args.data_dir / dataset if args.data_dir else None
        if path is not None and path.is_dir():
            # These small source datasets are checksum-validated workflow
            # artifacts. In-memory loading prevents Dataset.map/filter from
            # publishing mutable cache files alongside their source shards.
            ds = load_from_disk(str(path), keep_in_memory=True)
        else:
            if args.data_dir and not args.prepare:
                raise FileNotFoundError(
                    f"Materialize the dataset before compute: {path}"
                )
            ds = load_dataset("csv", data_files=url, split="train")
            if dataset == "ba-sascorer":
                ds = ds.map(lambda x: {"is_hard": x["accessibility"] == "hs"})
                ds = ds.select_columns(["smiles", "is_hard"])
            else:
                ds = ds.select_columns(["SMILES", "meanComplexity", "stdevComplexity"])
                ds = ds.rename_column("SMILES", "smiles")
                ds = ds.map(lambda x: {"is_complex": x["meanComplexity"] > 2.85})
            if path is not None:
                ds.save_to_disk(str(path))
        if args.prepare:
            continue
        if "row_id" not in ds.column_names:
            ds = ds.add_column("row_id", list(range(len(ds))))
        target = "is_hard" if dataset == "ba-sascorer" else "is_complex"
        stats = evaluate_dataset(metrics, ds, target=target)
        output = Path(args.output_format.format(dataset=dataset))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    if args.prepare:
        if args.data_dir is None:
            p.error("--prepare requires --data-dir")
        (args.data_dir / "receipt.json").write_text(
            json.dumps(sources, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
