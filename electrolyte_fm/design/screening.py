from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import BRICS as brics
from lightning.fabric import Fabric
from datasets import IterableDataset
import pandas as pd

from ..models.prod_finetune import MISTMultiTask


class OracleCritic(nn.Module):
    def __init__(self, oracle: nn.Module, critic):
        super().__init__()
        self.oracle = oracle
        self.critic = critic

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        y = self.oracle(input_ids, attention_mask=attention_mask)
        score = self.critic(y)
        return score


class QuadrantCritic:
    def __init__(self, ranges: dict[str, tuple[float, float]], channels: list[str]):
        self.ranges = ranges
        self.channels = channels

    def __call__(self, y: torch.Tensor):
        out = torch.zeros(y.shape[:-2], dtype=y.dtype, device=y.device)
        for idx, chn in enumerate(self.channels):
            if chn in self.ranges:
                lb, ub = self.ranges[chn]
                y_chn = y[..., idx]
                out &= (lb < y_chn & y_chn < ub).all(-1)

        return out


def fragment_molecules(smiles: str | list[str]) -> set[str]:
    if isinstance(smiles, str):
        smiles = [smiles]

    fragments = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fragments.update(brics.BRICSDecompose(mol))

    return fragments


def brics_dataset(
    fragments: set[str], tokenizer, batch_size=256, prefetch_factor=2, num_workers=0
):
    mol_fragments = [Chem.MolFromSmiles(x) for x in fragments]

    def gen():
        for mol in brics.BRICSBuild(mol_fragments):
            yield {"smi": Chem.MolToSmiles(mol)}

    ds = IterableDataset.from_generator(gen)
    ds = ds.map(tokenizer, input_columns="smi", batched=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )


def generate():
    root_dir = Path(__file__).parent.parent.parent
    ref_molecules = pd.read_csv(root_dir.joinpath("opt/design/electrolytes.csv"))["smi"]
    print(ref_molecules)

    fragments = fragment_molecules(ref_molecules)
    print(fragments)

    oracle = MISTMultiTask.from_pretrained(
        root_dir.joinpath("models/electrolyte-solvent")
    )
    critic = QuadrantCritic({"bp": (60, None), "mp": (-100, None), "fp": (60, None)})
    model = OracleCritic(oracle, critic)
    ds = brics_dataset(fragments, tokenizer=oracle.tokenizer)

    fabric = Fabric()
    fabric.launch()
    model = fabric.setup(model)
    dl = fabric.setup_dataloaders(ds)

    for batch in dl:
        print(batch)
        y = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        score = critic(y)


if __name__ == "__main__":
    generate()
