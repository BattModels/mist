"""Uneven wall-clock termination must not strand another rank in a reduction."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Panel(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.channels = []

    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1), torch.zeros(1, dtype=torch.bool)


class _Fabric:
    device = torch.device("cpu")

    def __init__(self, rank):
        self.global_rank = rank

    def all_reduce(self, value, reduce_op):
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        dist.all_reduce(tensor, op={"min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}[reduce_op])
        return tensor if isinstance(value, torch.Tensor) else tensor.tolist()


def _worker(rank, rendezvous):
    from src import generate as generation

    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        generation.CriticPanel = _Panel
        # Rank 0 finishes before rank 1 reaches its periodic 64-batch reduction.
        batches = [
            {"input_ids": torch.ones(1, 1, dtype=torch.long),
             "attention_mask": torch.ones(1, 1), "smi": [f"{rank}-{index}"]}
            for index in range(1 if rank == 0 else 65)
        ]
        assert list(generation.generate(_Fabric(rank), [object()], batches)) == []
        # Both ranks must have left generation before the final trace barrier.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_uneven_rank_completion(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
