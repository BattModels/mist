import gzip
from pathlib import Path
from typing import Optional, List, Callable
import json

import torch
from torch.nn import functional as F
from datasets import Dataset, load_dataset
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from sklearn.metrics import roc_auc_score, r2_score
from syba.syba import SybaClassifier
from transformers import AutoModelForMaskedLM, DataCollatorWithPadding

from electrolyte_fm.data_modules.utils import MolEncoding, encode_molecules
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer
from electrolyte_fm.utils.cache import cached_download, extract_file

from vendor.scscore.scscore import SCScorer


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
            score = score / (~batch["special_tokens_mask"].bool()).sum(-1)
        return score

    def score(self, smiles: List[str]) -> List[float]:
        batch = self.tokenizer(smiles, return_special_tokens_mask=True)
        batch = self.collate_fn(batch).to(self.encoder.device)
        with torch.no_grad():
            return self.forward(batch).to("cpu")

    @classmethod
    def from_checkpoint(cls, ckpt: str, **kwargs):
        encoder = DeepSpeedMixin.load(ckpt).model
        tokenizer = load_tokenizer(ckpt)
        return cls(encoder, tokenizer, **kwargs)

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        encoder = AutoModelForMaskedLM.from_pretrained(name_or_path)
        tokenizer = load_tokenizer(name_or_path)
        return cls(encoder, tokenizer, **kwargs)


def bascore(smiles: str) -> Optional[float]:
    pass


if __name__ == "__main__":
    # Setup dataset
    ds = load_dataset(
        "csv",
        name="BA-SAScore",
        data_files=[
            "https://raw.githubusercontent.com/snu-micc/BR-SAScore/refs/heads/main/data/test_set.csv"
        ],
    )
    ds = ds["train"].take(10)
    ds = ds.map(lambda x: {"is_hard": x["accessibility"] == "hs"}, batched=False)
    ds = ds.select_columns(["smiles", "is_hard"])


def evaluate_dataset(
    metrics: dict[str, Callable | str],
    ds: Dataset,
    target: str | None,
    smi_column: str = "smiles",
):
    ds = ds.rename_column(smi_column, "smiles")
    ds = encode_molecules(
        ds,
        "smiles",
        output_column="smiles-keukle",
        encoding=MolEncoding.KEUKLE_SMILES,
    )
    ds = encode_molecules(
        ds,
        "smiles",
        output_column="smiles-canonical",
        encoding=MolEncoding.CANONICAL_SMILES,
    )

    stats = {"auroc": {}} if target is not None else {}
    for name, metric in metrics.items():
        if isinstance(metric, str):
            model = SynthAccessFM.from_pretrained(metric)  # .to("cuda")
            continue
            for encoding in ["smiles", "smiles-keukle", "smiles-canonical"]:
                name = f"{metric}-{encoding}"
                ds = ds.map(
                    lambda x: {name: model.score(x[encoding])},
                    batched=True,
                    batch_size=64,
                )
                if target:
                    stats["auroc"][name] = roc_auc_score(ds[target], ds[name])

        # else:
        #     ds = ds.map(
        #         lambda x: {name: scorer(x)}, input_columns=smi_column, batched=False
        #     )
        #     if target:
        #         stats["auroc"][name] = roc_auc_score(ds[target], ds[name])

    # Score Molecules
    df = ds.to_pandas()
    stats["scores"] = df.to_dict("records")

    return stats


if __name__ == "__main__":
    syba = syba_scorer()
    scscore = SCScorer()
    metrics = {
        "sascore": sascore,
        "syba": lambda smi: -syba.predict(smi),
        "scscore": lambda smi: scscore.get_score_from_smi(smi),
        "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
        "molformer": "ibm/MoLFormer-XL-both-10pct",
        "mist-28znv46w": "models/mist-28znv46w",
        "mist-ti624ev1": "models/mist-ti624ev1",
    }

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
    with open("ba-sascorer.json", "w") as fid:
        json.dump(stats, fid, indent=4)

    # Modeling a Crowdsourced Definition of Molecular Complexity
    ds = load_dataset(
        "csv",
        name="figshare_3917065",
        data_files=["https://ndownloader.figstatic.com/files/3917065"],
    )
    ds = ds["train"]
    ds = ds.select_columns(["SMILES", "meanComplexity", "stdevComplexity"])
    stats = evaluate_dataset(metrics, ds, smi_column="SMILES")
    with open("crowdsourced.json") as fid:
        json.dump(stats, fid, indent=4)

    # Assembly Index
    ds = load_dataset(
        "csv",
        name="combined_results_ms",
        data_files=[
            "https://github.com/anoushka2000/electrolyte-fm/raw/refs/heads/assembly_index/opt/assembly_index/data/combined_results_ms.csv"
        ],
    )
    ds = ds.select_columns(["SMILES", "MA", "MA (est.)", "MA (mean est.)"])
    ds = ds.map(lambda x: {"biosignature": x["MA"] > 15}, batched=False)
    stats = evaluate_dataset(metrics, ds, smi_column="smiles")
    with open("assembly_index.json") as fid:
        json.dump(stats, fid, indent=4)
