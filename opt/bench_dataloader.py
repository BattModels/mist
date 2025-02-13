import json
from dataclasses import dataclass
from time import perf_counter
from typing import Optional
from random import randint

import _jsonnet as jsonnet
import numpy as np
from lightning import LightningDataModule

from electrolyte_fm.data_modules import MolNetDataModule, tmQMDataModule
from submit.utils import dict_product


@dataclass
class DummyTrainer:
    global_rank: int = 0
    world_size: int = 1


def benchmark_dataloader(
    dm: LightningDataModule,
    stage: str = "fit",
    limit_batches: int = 100,
    trials: int = 10,
    p=0.975,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
):
    if world_size is not None:
        rank = rank or 0
        dm.trainer = DummyTrainer(global_rank=rank, world_size=world_size)  # ignore

    dm.prepare_data()
    dm.setup(stage)

    time = []
    batch_size = dm.train_dataloader().batch_size
    for _ in range(trials):
        start = perf_counter()
        for idx, batch in enumerate(dm.train_dataloader()):
            if idx >= limit_batches:
                break
        time.append(perf_counter() - start)

    time = np.array(time)
    samples = limit_batches * batch_size
    return {
        "time_per_batch_mean": (time / limit_batches).mean(),
        "time_per_batch_std": (time / limit_batches).std(),
        "time_per_batch_upper": np.quantile(time / limit_batches, p),
        "samples_per_sec_mean": (samples / time).mean(),
        "samples_per_sec_std": (samples / time).std(),
        "samples_per_sec_lower": np.quantile(samples / time, 1 - p),
    }


if __name__ == "__main__":
    dms = []
    molnet_config = json.loads(
        jsonnet.evaluate_file("submit/moleculenet_tasks.libsonnet")
    )
    sweep = {
        "dataset": ["qm9", "sider"],
        "batch_size": [64, 128],
        "num_workers": [0, 1, 4, 8, 16, 32, 64],
        "encoding": ["smiles", "smiles-kekule"],
        "prefetch_factor": [None, 1, 4, 8, 16],
    }
    for c in dict_product(sweep):
        if c["num_workers"] == 0 and c["prefetch_factor"] is not None:
            continue

        if c["dataset"] != "tmQM":
            c["module"] = MolNetDataModule(
                name=c["dataset"],
                batch_size=c["batch_size"],
                num_workers=c["num_workers"],
                encoding=c["encoding"],
                prefetch_factor=c["prefetch_factor"],
                target_columns=molnet_config[c["dataset"]]["target_columns"],
            )

        dms.append(c)

    results = []
    for case in dms:
        dm = case.pop("module")
        print("benchmarking: ", case, end="")
        r = benchmark_dataloader(dm)
        print(
            f" {r['samples_per_sec_mean']:.3e} ± {r['samples_per_sec_std']:.3e} samples/sec"
        )
        results.append({**case, **r})

    with open("results.json", "w") as fid:
        json.dump(results, fid, indent=4)
