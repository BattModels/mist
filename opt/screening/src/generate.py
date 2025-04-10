import time
import logging
import itertools

import torch
from lightning.fabric import Fabric
from torch import nn


from electrolyte_fm.models.prod_finetune import MISTFinetuned
from .fasmifra import dataloader as fasmifra_dataloader
from .utils import RateLimitedAdapter


logger = RateLimitedAdapter(logging.getLogger(__name__), min_interval=30)


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
        for chn in channels:
            if chn in limits:
                lb, ub = limits[chn]
            else:
                lb, ub = None, None
            lower.append(lb or -torch.inf)
            upper.append(ub or torch.inf)

        self.register_buffer("lower", torch.tensor(lower).view(1, -1))
        self.register_buffer("upper", torch.tensor(upper).view(1, -1))

    def __call__(self, y: torch.Tensor):
        return ((self.lower < y) & (y < self.upper)).all(-1)


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
            input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device
        )
        y = []
        for critic in self.critics:
            yc, score = critic(input_ids, attention_mask=attention_mask)
            net_score &= score
            y.append(yc)

        return torch.cat(y, dim=-1), net_score


def randomized_dataloader(
    fabric: Fabric,
    ref_mol_file,
    tokenizer,
    batch_size: int = 128,
    encoding=None,
    epoch: int = 1024,
):
    while True:
        logger.info("Reloading dataloader on rank %d", fabric.global_rank)
        dl = fasmifra_dataloader(
            ref_mol_file,
            tokenizer,
            batch_size=batch_size,
            encoding=encoding,
            fabric=fabric,
        )
        yield from itertools.islice(dl, epoch)


def generate(fabric, critics, ref_mol_file, encoding: str | None = None):
    assert len(critics) > 0, "No critics provided"

    # Setup Critics
    panel = CriticPanel(critics).to(fabric.device, dtype=torch.bfloat16).eval()
    panel = torch.compile(panel, dynamic=True, fullgraph=True)

    # Setup Timing
    batch_time = 0.0
    time_exp = 0.95
    n_passing = 0
    n_evaluated = 0
    start_time = time.perf_counter()

    # Reshuffling Dataloader
    dl = randomized_dataloader(
        fabric,
        ref_mol_file,
        critics[0].oracle.tokenizer,
        encoding=encoding,
        batch_size=256,
    )

    # Generate molecules
    for batch in dl:
        start_batch = time.perf_counter()
        n_evaluated += batch["input_ids"].shape[0]

        with torch.inference_mode():
            y, net_score = panel(
                batch["input_ids"].to(fabric.device),
                attention_mask=batch["attention_mask"].to(fabric.device),
            )

        # Track performance metrics
        net_time = time.perf_counter() - start_time
        batch_time = (
            time_exp * (time.perf_counter() - start_batch) + (1 - time_exp) * batch_time
        )
        batch_passing = net_score.count_nonzero().item()
        n_passing += batch_passing
        logging.info(
            {
                "generated": n_passing,
                "evaluated": n_evaluated,
                "net_throughput": n_passing / net_time,
                "yield": n_passing / n_evaluated,
                "eval_throughput": n_evaluated / net_time,
                "current_throughput": batch_passing / batch_time,
            }
        )

        # Yield Passing Molecules
        if net_score.any():
            with torch.cuda.nvtx.range("yield_passing"):
                smiles: list[str] = [
                    smi for smi, s in zip(batch["smi"], net_score.tolist()) if s
                ]
                y = y[net_score].view(-1, y.shape[-1]).tolist()
                for smi, y in zip(smiles, y):
                    yield {"smi": smi, **{k: v for k, v in zip(panel.channels, y)}}

    logging.info("Rank %d: Finished", fabric.global_rank)
    return None
