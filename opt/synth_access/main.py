import argparse
import functools
import gzip
import json
import logging
import multiprocessing
from os import environ
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Optional

from sklearn.utils.multiclass import attach_unique
import torch
import accelerate
from assembly_theory import molecular_assembly
from BRSAScore import SAScorer as BRSAScorer
from datasets import Dataset, load_dataset
from rdkit import Chem, rdBase
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.Descriptors import MolWt
from sklearn.metrics import roc_auc_score
from syba.syba import SybaClassifier
from smirk import SmirkTokenizerFast
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding
from vendor.scscore.scscore import SCScorer

from electrolyte_fm.data_modules.utils import MolEncoding, encode_molecules
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.cache import cached_download, extract_file
from electrolyte_fm.utils.tokenizer import load_tokenizer

# Suppress DeprecationWarnings for MorganGenerator
rdBase.DisableLog("rdApp.warning")

logging.basicConfig(level=logging.INFO)

NUM_PROCS = int(environ.get("SLURM_CPUS_PER_TASK", 4))


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target(q, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    q.put(result)
                except Exception as e:
                    q.put(e)

            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=target, args=(q, *args), kwargs=kwargs)
            p.start()
            p.join(seconds)

            if p.is_alive():
                p.terminate()
                p.join()
                logging.warning(
                    f"Function '{func.__name__}' timed out after {seconds} seconds with args={args}, kwargs={kwargs}"
                )
                return None

            result = q.get()
            if isinstance(result, Exception):
                raise result
            return result

        return wrapper

    return decorator


def sascore(smiles: str) -> Optional[float]:
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


class SynthAccessFM(torch.nn.Module):
    def __init__(self, encoder, tokenizer, per_token: bool = False):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.per_token = per_token

        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def forward(self, batch):
        logits = self.encoder(
            batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        B = batch["input_ids"].shape[0]
        V = logits.shape[-1]

        labels = (
            batch["input_ids"]
            .detach()
            .masked_fill(batch["special_tokens_mask"].bool(), -100)
        )

        score = (
            F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction="none")
            .reshape(B, -1)
            .sum(-1)
        )
        if self.per_token:
            score = score / batch["attention_mask"].sum(-1)
        return score

    def score(self, smiles: List[str]) -> List[float]:
        batch = self.tokenizer(smiles, return_special_tokens_mask=True)
        batch = self.collate_fn(batch).to(self.encoder.device)
        with torch.inference_mode():
            return self.forward(batch).to("cpu")

    @classmethod
    def from_checkpoint(cls, ckpt: str, **kwargs):
        encoder = DeepSpeedMixin.load(ckpt).model
        tokenizer = load_tokenizer(ckpt)
        return cls(encoder, tokenizer, **kwargs)

    @classmethod
    def from_pretrained(cls, name_or_path: str, dtype=None, **kwargs):
        encoder = AutoModelForMaskedLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = load_tokenizer(name_or_path)
        return cls(encoder, tokenizer, **kwargs)

    @classmethod
    def from_untrained(cls, name_or_path: str, dtype=None, **kwargs):
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        encoder = AutoModelForMaskedLM.from_config(config)
        tokenizer = load_tokenizer(name_or_path)
        return cls(encoder, tokenizer, **kwargs)


@timeout(30)  # molecular_assembly (v0.2.0) timeout flag doesn't timeout
def molecular_assembly_timeout(smi: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(smi)
    will_error = [
        "CC(C)(C)OC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",  # Panics in isomorphism.rs
    ]
    if smi in will_error:
        return None
    try:
        return molecular_assembly(mol)
    except Exception as e:
        # Catch panics
        logging.error("molecular assembly threw an error for %s: %s", smi, e)
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

    raise RuntimeError("Failed to find a batch size that doesn't cause an OOM")


def eval_fm_model(
    metric_name: str, model: SynthAccessFM, ds: Dataset, target: str | None = None
):
    model = model.to("cuda")
    model = model.eval()
    runtime = dict()
    auroc_stats = dict()
    for encoding in ["smiles", "smiles-kekule", "smiles-canonical"]:
        for per_token in [False, True]:
            name = f"{metric_name}-{encoding}"
            if per_token:
                name += "-per-token"

            start = perf_counter()
            ds = map_batchsize_finder(
                ds,
                lambda x: {name: model.score(x)},
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

    stats = {"auroc": {}} if target is not None else {}
    stats["time"] = {}
    for name, metric in metrics.items():
        if isinstance(metric, str):
            for suffix, init_model in [
                ("", SynthAccessFM.from_pretrained),
                ("-untrained", SynthAccessFM.from_untrained),
            ]:
                ds, metric_runtime, metric_auroc = eval_fm_model(
                    metric + suffix,
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
                lambda x: {name: metric(x)},
                input_columns=input_columns,
                batched=False,
                desc=name,
                num_proc=NUM_PROCS,
            )
            stats["time"][name] = perf_counter() - start
            if target:
                ds_auroc = ds.filter(
                    lambda x: x is not None, batched=False, input_columns=name
                )
                stats["auroc"][name] = roc_auc_score(ds_auroc[target], ds_auroc[name])

    # Score Molecules
    df = ds.to_pandas()
    stats["scores"] = df.to_dict("records")

    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", type=str, action="append", default=None)
    p.add_argument("--skip-metric", type=str, action="append", default=None)
    p.add_argument("--output-format", type=str, default="{dataset}.json")
    args = p.parse_args()

    syba = syba_scorer()
    scscore = SCScorer().restore()
    scscore.restore()
    ba_sascorer = BRSAScorer()
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

    for file in Path("models").iterdir():
        if file.is_dir():
            metrics[file.name] = str(file)

    # Uncomment to just run assembly-index
    if args.metric is not None:
        metrics = {metric: metrics[metric] for metric in args.metric}

    if args.skip_metric is not None:
        for metric in args.skip_metric:
            metrics.pop(metric, None)
    logging.info("Evaluating: %s", ",".join(metrics.keys()))

    # BA-SAScore's Dataset
    ds = load_dataset(
        "csv",
        name="BA-SAScore",
        data_files=[
            "https://raw.githubusercontent.com/snu-micc/BR-SAScore/refs/heads/main/data/test_set.csv"
        ],
    )
    ds = ds["train"]
    ds = ds.map(lambda x: {"is_hard": x["accessibility"] == "hs"}, batched=False)
    ds = ds.select_columns(["smiles", "is_hard"])

    stats = evaluate_dataset(metrics, ds, target="is_hard", smi_column="smiles")
    with open(args.output_format.format(dataset="ba-sascorer"), "w") as fid:
        json.dump(stats, fid, indent=4)

    # Modeling a Crowdsourced Definition of Molecular Complexity
    ds = load_dataset(
        "csv",
        name="figshare_3917065",
        data_files=["https://ndownloader.figstatic.com/files/3917065"],
    )
    ds = ds["train"]
    ds = ds.select_columns(["SMILES", "meanComplexity", "stdevComplexity"])
    ds = ds.map(lambda x: {"is_complex": x["meanComplexity"] > 2.85}, batched=False)
    stats = evaluate_dataset(metrics, ds, target="is_complex", smi_column="SMILES")
    with open(args.output_format.format(dataset="crowdsourced"), "w") as fid:
        json.dump(stats, fid, indent=4)
