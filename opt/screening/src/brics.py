import itertools
import math
from copy import deepcopy
import torch
from rdkit import Chem
from rdkit.Chem import BRICS as brics
from transformers import DataCollatorWithPadding


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


class BricsDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fragments: list[str],
        tokenizer,
        max_depth: int = 2,
        num_fragments: int = 8,
        brics_batch_size: int = 64,
    ):
        super().__init__()
        self.fragments = list(fragments)
        self.max_depth = int(max_depth)
        self.tokenizer = tokenizer
        self.brics_batch_size = brics_batch_size
        self.num_fragments = max(len(self.fragments), num_fragments)
        self.token_collator = DataCollatorWithPadding(tokenizer)

    @property
    def seeds(self):
        frags = self.fragments
        if worker_info := torch.utils.data.get_worker_info():
            per_worker = int(math.ceil(len(frags) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(frags))
            return frags[iter_start:iter_end]
        return frags

    def collate_fn(self, batch):
        smi = [x.pop("smi") for x in batch]
        out = self.token_collator(batch)
        out["smi"] = smi
        return out

    def __iter__(self) -> None:
        seeds = [Chem.MolFromSmiles(x) for x in self.seeds]
        mol_fragments = [Chem.MolFromSmiles(x) for x in self.fragments]
        tokenizer = deepcopy(self.tokenizer)
        while True:
            # maxDepth = random.randint(1, self.max_depth)
            seed = random.choice(seeds)
            # fragments = random.choices(mol_fragments, k=self.num_fragments)

            for mols in itertools.batched(
                itertools.islice(
                    brics.BRICSBuild(
                        mol_fragments, maxDepth=self.max_depth, seeds=[seed]
                    ),
                    64,
                ),
                self.brics_batch_size,
            ):
                smiles = [Chem.MolToSmiles(x) for x in mols]
                tokens = tokenizer(smiles)
                for smi, input_ids, attention_mask in zip(
                    smiles, tokens["input_ids"], tokens["attention_mask"]
                ):
                    yield {
                        "smi": smi,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }

    def dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=False, **kwargs)
