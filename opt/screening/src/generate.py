import logging
import time

import torch
from lightning.fabric import Fabric
from torch import nn

from electrolyte_fm.models.prod_finetune import MISTFinetuned
from src.hyperloglog import HyperLogLogSet

from .utils import RateLimitedAdapter, configure_logging


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
        return y, score

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        limits: dict,
        model_cls=MISTFinetuned,
    ):
        oracle = model_cls.from_pretrained(save_directory)
        critic = QuadrantCritic(
            limits, channels=[chn["name"] for chn in oracle.channels]
        )
        return cls(oracle, critic)


class QuadrantCritic(nn.Module):
    def __init__(self, limits: dict[str, tuple[float, float]], channels: list[str]):
        super().__init__()
        lower = []
        upper = []
        assert limits.keys() <= set(
            channels
        ), f"limits must be a subset of channels: {limits.keys()} ⊆ {channels}"
        for chn in channels:
            if chn in limits:
                lb, ub = limits[chn]
            else:
                lb, ub = None, None
            lower.append(-torch.inf if lb is None else lb)
            upper.append(torch.inf if ub is None else ub)

        self.register_buffer("lower", torch.tensor(lower).view(1, -1))
        self.register_buffer("upper", torch.tensor(upper).view(1, -1))

    @property
    def active_channels(self):
        return ~(self.lower.isinf() & self.upper.isinf()).view(-1)

    def __call__(self, y: torch.Tensor):
        return (self.lower < y) & (y < self.upper)


class CriticPanel(nn.Module):
    def __init__(self, critics: list[OracleCritic]):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        channels = []
        for critic in critics:
            channels.extend([chn["name"] for chn in critic.oracle.channels])
        self.channels: list[str] = channels

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        net_score = torch.ones(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        y = []
        for critic in self.critics:
            yc, score = critic(input_ids, attention_mask=attention_mask)
            net_score &= score.all(-1)
            y.append(yc)

        return torch.cat(y, dim=-1), net_score


def generate(fabric: Fabric, critics, mol_dataloader):
    assert len(critics) > 0, "No critics provided"

    # Setup Critics
    panel = CriticPanel(critics).to(fabric.device, dtype=torch.bfloat16).eval()
    panel = torch.compile(panel, dynamic=True, fullgraph=True)

    # Setup Timing
    batch_time = 0.0
    time_exp = 0.95
    n_passing = 0
    n_evaluated = 0
    generated_molecules = HyperLogLogSet()
    passing_molecules = HyperLogLogSet()
    start_time = time.perf_counter()
    world_size = fabric.world_size
    last_sync = 0
    log_rate = 1000

    # Generate molecules
    for idx, batch in enumerate(mol_dataloader):
        start_batch = time.perf_counter()
        n_evaluated += batch["input_ids"].shape[0]

        with torch.inference_mode():
            y, net_score = panel(
                batch["input_ids"].to(fabric.device),
                attention_mask=batch["attention_mask"].to(fabric.device),
            )

        # Track performance metrics
        net_time = time.perf_counter() - start_time
        generated_molecules.extend(batch["smi"])
        batch_time = (
            time_exp * (time.perf_counter() - start_batch) + (1 - time_exp) * batch_time
        )
        batch_passing = net_score.count_nonzero().item()
        n_passing += batch_passing
        n_passing_world = len(passing_molecules)
        if n_passing_world > 0:
            time_per_passing = (net_time * world_size) / n_passing_world
        else:
            time_per_passing = None

        if idx % log_rate == 0:
            logging.info(
                {
                    "passing_rank": n_passing,
                    "passing_world": n_passing_world,
                    "unique_molecules_world": len(generated_molecules),
                    "evaluated_rank": n_evaluated,
                    "net_throughput_rank": n_passing / net_time,
                    "yield_rank": n_passing / n_evaluated,
                    "eval_throughput_rank": n_evaluated / net_time,
                    "eval_throughput_world": len(generated_molecules)
                    / (net_time * world_size),
                    "current_throughput_rank": batch_passing / batch_time,
                    "time_per_passing_world": time_per_passing,
                }
            )

        # Yield Passing Molecules
        if net_score.any():
            with torch.cuda.nvtx.range("yield_passing"):
                smiles: list[str] = [
                    smi for smi, s in zip(batch["smi"], net_score.tolist()) if s
                ]
                passing_molecules.extend(smiles)
                y = y[net_score].view(-1, y.shape[-1]).tolist()
                for smi, y in zip(smiles, y):
                    yield {"smi": smi, **{k: v for k, v in zip(panel.channels, y)}}

        # Synchronize generated cardinality
        last_sync += 1
        if last_sync >= 64:
            logging.debug("Synchronizing generated cardinality")
            generated_molecules.reduce(fabric)
            passing_molecules.reduce(fabric)
            last_sync = 0

    logging.info("Rank %d: Finished", fabric.global_rank)
    return None
