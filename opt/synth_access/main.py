import gzip
from pathlib import Path
from typing import Optional, List

import torch
from torch.nn import functional as F
from datasets import load_dataset
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from sklearn.feature_selection import r_regression
from sklearn.metrics import roc_auc_score
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

    mol = Chem.MolFromSmiles(smiles)


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
    def __init__(self, encoder: str, tokenizer: str = None, per_token=False):
        super().__init__()
        if Path(encoder).exists():
            self.encoder = DeepSpeedMixin.load(encoder).model
        else:
            from transformers import AutoModel

            self.encoder = AutoModelForMaskedLM.from_pretrained(
                encoder, trust_remote_code=True
            )

        self.tokenizer = load_tokenizer(tokenizer or encoder)
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)
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
    ds = ds.map(lambda x: {"is_hard": x["accessibility"] == "hs"}, batched=False)
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

    scorers = []

    # SAScore
    ds = ds.map(lambda x: {"sascore": sascore(x["smiles"])}, batched=False)
    scorers.append("sascore")

    # SYBA
    # Flip sign to get positive = hard to synthesize
    syba = syba_scorer()
    ds = ds.map(lambda x: {"syba": -syba.predict(x["smiles"])}, batched=False)
    scorers.append("syba")

    # SCScore
    scscore = SCScorer()
    scscore.restore()
    ds = ds.map(
        lambda x: {"scscore": scscore.get_score_from_smi(x["smiles"])[1]}, batched=False
    )
    scorers.append("scscore")

    # FM
    ckpts = {
        "molformer": "ibm/MoLFormer-XL-both-10pct",
        "mist-apt8adhv": "/home/awadell/scratch/mist_ckpt/apt8adhv/checkpoints/epoch=0-step=6100-val_loss=0.02.ckpt",
        "mist-jico5kgk": "/home/awadell/scratch/mist_ckpt/jico5kgk/checkpoints/epoch=0-step=12800-val_loss=0.01.ckpt",
    }
    for id, ckpt_path in ckpts.items():
        model = SynthAccessFM(ckpt_path).to("cuda")
        for encoding in ["smiles", "smiles-keukle", "smiles-canonical"]:
            name = f"{id}-{encoding}"
            ds = ds.map(
                lambda x: {name: model.score(x[encoding])},
                batched=True,
                batch_size=64,
            )
            scorers.append(name)
        del model

    # Score Molecules
    df = ds["train"].to_pandas()
    df.to_csv("scores.csv")

    # Score Predictions
    stats = {}
    stats["auroc"] = {}
    for ref in scorers:
        stats["auroc"][ref] = roc_auc_score(df["is_hard"], df[ref])

    for ref in scorers:
        print(f"{ref}: {stats['auroc'][ref]:.3f}")
