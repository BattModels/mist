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
from transformers import AutoModelForMaskedLM, DataCollatorWithPadding

from electrolyte_fm.data_modules.utils import MolEncoding, encode_molecules
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer
from electrolyte_fm.utils.cache import cached_download, extract_file





class AssemblyIndexFM(torch.nn.Module):
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



if __name__ == "__main__":
    # Setup dataset
    ds = load_dataset(
        "csv",
        name="Assembly",
        data_files=[
            "./data/combined_results_ms.csv"
        ],
    )
    ds = ds.map(lambda x: {"biosignatures": x["MA"] > 15}, batched=False)
    ds = encode_molecules(
        ds,
        "SMILES",
        output_column="smiles-keukle",
        encoding=MolEncoding.KEUKLE_SMILES,
    )
    ds = encode_molecules(
        ds,
        "SMILES",
        output_column="smiles-canonical",
        encoding=MolEncoding.CANONICAL_SMILES,
    )

    scorers = []

    scorers.append("MA (MW est.)")
    scorers.append("MA (mean est.)")
    
    # FM
    ckpts = {
        "molformer": "ibm/MoLFormer-XL-both-10pct",
        "mist-xzzorelb": "/projects/bcuf/abhutani/mist_pretrained/xzzorelb/checkpoints/last.ckpt",
        "mist-icfja3kx": "/projects/bcuf/abhutani/mist_pretrained/icfja3kx/checkpoints/last.ckpt",
        "mist-atleto2u": "/projects/bcuf/abhutani/mist_pretrained/atleto2u/checkpoints/last.ckpt",
        "mist-n2dkcidc": "/projects/bcuf/abhutani/mist_pretrained/n2dkcidc/checkpoints/last.ckpt",
        "mist-apt8adhv": "/projects/bcuf/abhutani/mist_pretrained/apt8adhv/checkpoints/last.ckpt",
    }
    for id, ckpt_path in ckpts.items():
        model = AssemblyIndexFM(ckpt_path).to("cuda")
        for encoding in [ "smiles-canonical"]:
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
        stats["auroc"][ref] = roc_auc_score(df["biosignatures"], df[ref])

    for ref in scorers:
        print(f"{ref}: {stats['auroc'][ref]:.3f}")
