import random
import sqlite3
import logging
from pathlib import Path
from typing import Iterator, Optional, Union

import torch
from lightning import Fabric
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from electrolyte_fm.data_modules.utils import MolEncoding
from electrolyte_fm.utils.tokenizer import load_tokenizer

from .fasmifra import FASMIFRA
from .hyperloglog import HyperLogLogSet


class FragmentDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fabric: Fabric,
        tokenizer,
        encoder=MolEncoding.KEKULE,
        seed: int = 42,
        increment_seed: bool = True,
        epoch_size: int = 1_000,
    ) -> None:
        self.fabric = fabric
        self.tokenizer = tokenizer
        self.encoder: MolEncoding = encoder
        self.seed: int = seed
        self.epoch_size = epoch_size
        self.increment_seed = increment_seed

    def fragments(self) -> Iterator[str]:
        raise NotImplementedError("Subclasses must implement fragments()")

    def __iter__(self) -> Iterator[dict[str, str]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed_step = 1
        else:
            seed_step = worker_info.num_workers
        self.seed = (self.seed + seed_step) % (1 << 32)
        self._fasmisfra = iter(
            FASMIFRA(self.fragments(), encoder=self.encoder, seed=self.seed)
        )
        self._epoch_count = 0
        return self

    def __next__(self):
        if self._epoch_count >= self.epoch_size:
            self.__iter__()
        self._epoch_count += 1
        return next(self._fasmisfra)

    def collate_fn(self, tokenizer):
        token_collate = DataCollatorWithPadding(tokenizer)

        def collate(batch):
            smi = [x.pop("smi") for x in batch]
            out = token_collate(tokenizer(smi))
            out["smi"] = smi
            return out

        return collate

    def dataloader(
        self,
        batch_size: int = 512,
        prefetch_factor: int = 4,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn(self.tokenizer),
            num_workers=8,
            prefetch_factor=prefetch_factor,
        )


class ShuffledFragmentDataset(FragmentDataset):
    def __init__(
        self, fabric: Fabric, tokenizer, ref_frag_file: Union[str, Path], **kwargs
    ):
        super().__init__(fabric, tokenizer, **kwargs)
        self.ref_frag_file = Path(ref_frag_file)

    def fragments(self) -> Iterator[str]:
        lines = None
        if self.fabric.is_global_zero:
            lines = self.ref_frag_file.read_text().splitlines()
            random.shuffle(lines)
        self.fabric.barrier("shard_fragments - broadcast")
        lines = self.fabric.broadcast(lines)

        lines = lines[self.fabric.global_rank :: self.fabric.world_size]
        for line in lines:
            yield line


class WithReplacementFragmentDataset(FragmentDataset):
    def __init__(
        self, fabric: Fabric, tokenizer, ref_frag_file: Union[str, Path], **kwargs
    ):
        super().__init__(fabric, tokenizer, **kwargs)
        self.ref_frag_file = Path(ref_frag_file)

    def fragments(self) -> Iterator[str]:
        p_pick = 1 / self.fabric.world_size
        with self.ref_frag_file.open("r", encoding="utf-8") as frags:
            for line in frags:
                if random.random() < p_pick:
                    yield line.strip()


class DatabaseFragmentDataset(FragmentDataset):
    def __init__(
        self,
        fabric: Fabric,
        tokenizer,
        db_path: Union[str, Path],
        n_fragments: int = 1000,
        ref_frag_file: Optional[Union[str, Path]] = None,
        limit_db_fragments: Optional[int] = None,
        limit_ref_fragments: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(fabric, tokenizer, **kwargs)
        self.db_path = db_path
        self.n_fragments = n_fragments

        conn = sqlite3.connect(db_path, timeout=30)
        self.max_id = conn.execute("SELECT MAX(id) FROM fragments").fetchone()[0]
        if limit_db_fragments:
            assert 0 < limit_db_fragments
            self.max_id = min(self.max_id, limit_db_fragments)
        logging.info(
            {"message": "found db fragments", "unique": self.max_id, "path": db_path}
        )

        # Get unique fragments from ref_frag_file, if provided
        ref_frags = set()
        if ref_frag_file is not None:
            with Path(ref_frag_file).open("r") as frags:
                for line in frags:
                    ref_frags.add(line.strip())
        if limit_ref_fragments:
            random.seed(42)
            ref_frags = random.sample(list(ref_frags), k=limit_ref_fragments)
        self.ref_fragments = list(ref_frags)

        logging.info(
            {
                "message": "found ref. fragments",
                "unique": len(self.ref_fragments),
                "path": ref_frag_file,
            }
        )

    def fragments(self) -> Iterator[str]:
        if self.ref_fragments:
            yield from self.ref_fragments

        conn = sqlite3.connect(self.db_path, timeout=30)
        remaining = self.n_fragments
        while remaining >= 1:
            k = min(remaining, 10_000)
            random_ids = random.sample(range(1, self.max_id), k=k)
            placeholders = ",".join("?" for _ in random_ids)
            cursor = conn.execute(
                f"SELECT fragment FROM fragments WHERE id IN ({placeholders})",
                random_ids,
            )
            for (pattern,) in cursor.fetchall():
                yield pattern
            remaining -= k
        conn.close()


if __name__ == "__main__":
    import logging
    from .hyperloglog import HyperLogLogSet

    logging.basicConfig(level=logging.INFO)

    ds = DatabaseFragmentDataset(
        Fabric(),
        load_tokenizer("smirk"),
        "zinc_fragment/fragments.sqlite",
        n_fragments=5_000,
        ref_frag_file="electrolytes.smi.frag",
        epoch_size=5_000,
    )

    hll = HyperLogLogSet()
    for batch in ds.dataloader():
        n_start = len(hll)
        hll.extend(batch["smi"])
        n_end = len(hll)
        logging.info(
            {
                "batch_size": len(batch["smi"]),
                "batch_new": n_end - n_start,
                "batch_novelty": (n_end - n_start) / len(batch["smi"]),
                "true_batch_new": len(set(batch["smi"])),
            }
        )
