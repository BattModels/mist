import logging
import time
from abc import ABC, abstractmethod

import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from transformers import AutoModel

from .hyperloglog import HyperLogLogSet
from .prod_finetune import MISTFinetuned, MISTMultiTask


class Critic(nn.Module, ABC):
    """
    Abstract base class for any “Critic.” Subclasses must implement
    `active_channels`, which returns a 1‐D boolean Tensor of length = C,
    where C = number of channels.  True means “that channel is active.”

    We override __repr__ (and __str__) so that printing any Critic object
    will show its class name + the active‐channels mask.
    """

    @property
    @abstractmethod
    def active_channels(self) -> Tensor:
        """
        Return a 1‐D boolean Tensor (length = num_channels) indicating
        which channels are active.  Subclasses must override this.
        """
        ...

    @abstractmethod
    def forward(self, y: Tensor) -> Tensor:
        """
        Given a tensor (..., C), return a tensor (...) indicating which entries are passing
        """
        ...


class OracleCritic(nn.Module):
    def __init__(self, oracle: nn.Module, critic: Critic):
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
        model_path: str,
        limits: dict | None = None,
        all_passing: bool | None = None,
        any_passing: bool | None = None,
        model_cls: str = "MISTFinetuned",
        **kwargs,
    ):
        oracle_cls = {
            "MISTFinetuned": MISTFinetuned,
            "MISTMultiTask": MISTMultiTask,
            "AutoModel": AutoModel,
        }.get(model_cls)
        oracle = oracle_cls.from_pretrained(model_path)
        if isinstance(oracle.channels[0], dict):
            channels = [chn["name"] for chn in oracle.channels]
        elif isinstance(oracle.channels[0], str):
            channels = oracle.channels
        if limits is not None:
            critic = QuadrantCritic.from_limits(limits, channels)
        elif all_passing is not None:
            critic = QuadrantCritic.from_all_passing(channels, **kwargs)
        elif any_passing is not None:
            critic = AnyCritic.from_any_passing(channels, **kwargs)
        else:
            raise RuntimeError("Unknown critic type")

        return cls(oracle, critic)


def logit_limits(
    channels: list[str],
    pass_positive: bool = True,
    flip_channels: dict[str, bool] | None = None,
):
    limits = dict()
    flip_channels = flip_channels or {}
    for chn in channels:
        flip_channels[chn] = flip_channels.get(chn, False)
        if (pass_positive and not flip_channels[chn]) or (
            not pass_positive and flip_channels[chn]
        ):
            limits[chn] = (0, None)
        else:
            limits[chn] = (None, 0)
    return limits


def limits_to_bounds(limits: dict[str, tuple[float, float]], channels: list[str]):
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
    return lower, upper


class QuadrantCritic(Critic):
    def __init__(self, lower: Tensor, upper: Tensor):
        super().__init__()
        self.register_buffer("lower", lower.view(1, -1))
        self.register_buffer("upper", upper.view(1, -1))

    @classmethod
    def from_limits(cls, limits: dict[str, tuple[float, float]], channels: list[str]):
        lower, upper = limits_to_bounds(limits, channels)
        return cls(torch.tensor(lower), torch.tensor(upper))

    @classmethod
    def from_all_passing(
        cls,
        channels: list[str],
        pass_positive: bool = True,
        flip_channels: dict[str, bool] | None = None,
    ):
        limits = logit_limits(channels, pass_positive, flip_channels)
        return cls.from_limits(limits, channels)

    @property
    def active_channels(self):
        return ~(self.lower.isinf() & self.upper.isinf()).view(-1)

    def forward(self, y: torch.Tensor):
        y = torch.atleast_2d(y)
        return ((self.lower < y) & (y < self.upper)).all(-1)


class AnyCritic(Critic):
    def __init__(self, lower: Tensor, upper: Tensor, mask: Tensor):
        super().__init__()
        self.register_buffer("lower", lower.view(1, -1))
        self.register_buffer("upper", upper.view(1, -1))
        self.register_buffer("mask", mask.to(dtype=bool).view(1, -1))

    @classmethod
    def from_any_passing(
        cls,
        channels: list[str],
        pass_positive: bool = True,
        flip_channels: dict[str, bool] | None = None,
        subset: list[str] | None = None,
    ):
        limits = logit_limits(channels, pass_positive, flip_channels)
        lower, upper = limits_to_bounds(limits, channels)
        subset_mask = []
        for chn in channels:
            if subset is None:
                subset_mask.append(True)
            else:
                subset_mask.append(chn in subset)
        return cls(torch.tensor(lower), torch.tensor(upper), torch.tensor(subset_mask))

    @property
    def active_channels(self):
        active_limits = ~(self.lower.isinf() & self.upper.isinf())
        return (active_limits & self.mask).view(-1)

    def forward(self, y: torch.Tensor):
        """Return True if any channel is active for the molecule"""
        y = torch.atleast_2d(y)
        return ((self.lower < y) & (y < self.upper) & (self.mask)).any(-1)


class CriticPanel(nn.Module):
    def __init__(self, critics: list[OracleCritic]):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        channels = []
        for critic in critics:
            if isinstance(critic.oracle.channels[0], dict):
                channels.extend([chn["name"] for chn in critic.oracle.channels])
            elif isinstance(critic.oracle.channels[0], str):
                channels.extend(critic.oracle.channels)
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
            net_score &= score
            y.append(yc)

        return torch.cat(y, dim=-1), net_score


def generate(fabric: Fabric, critics, mol_dataloader):
    assert len(critics) > 0, "No critics provided"

    # Setup Critics
    panel = CriticPanel(critics).to(fabric.device).eval()

    # Setup Timing
    batch_time = 0.0
    time_exp = 0.95
    n_passing = 0
    n_evaluated = 0
    generated_molecules = HyperLogLogSet()
    passing_molecules = HyperLogLogSet()
    start_time = time.perf_counter()
    last_sync = 0

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

        if idx % 64 == 0:
            logging.info(
                {
                    "passing_rank": n_passing,
                    "evaluated_rank": n_evaluated,
                    "net_throughput_rank": n_passing / net_time,
                    "yield_rank": n_passing / n_evaluated,
                    "eval_throughput_rank": n_evaluated / net_time,
                    "current_throughput_rank": batch_passing / batch_time,
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
            generated_molecules, passing_molecules = sync_cardinality(
                fabric, generated_molecules, passing_molecules
            )

            last_sync = 0
    logging.info("Rank %d: Finished", fabric.global_rank)
    sync_cardinality(fabric, generated_molecules, passing_molecules)

    return None


def sync_cardinality(fabric, generated_molecules, passing_molecules):
    logging.debug("Synchronizing generated cardinality")
    generated_molecules = generated_molecules.reduce(fabric)
    passing_molecules = passing_molecules.reduce(fabric)
    logging.info(
        {
            "n_passing_world": len(passing_molecules),
            "unique_molecules_world": len(generated_molecules),
        }
    )
    return generated_molecules, passing_molecules
