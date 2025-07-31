from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch.optim import LBFGS
from difftopk import DiffTopkNet
from electrolyte_fm.models.model_utils import masked_mean_pool
from smirk import SmirkTokenizerFast
import torch.nn.functional as F
import sys

from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsLightningModel,
    ExcessPhysicsModel,
)
sys.path.append(
    Path(__file__).absolute().parent.parent.joinpath("python", "excess.py")
)


class ExcessOptimizer:
    def __init__(
        self, pretrained_ckpt: str, temperature: float,
    ) -> None:
        self.pretrained_ckpt = pretrained_ckpt
        self.model = self._load_model(self.pretrained_ckpt)
        self.tokenizer = SmirkTokenizerFast()
        self.temperature = temperature

        self.sorter = DiffTopkNet(
            sorting_network_type="bitonic",
            size=len(self.inventory),
            k=2,
            sparse=False,
            device=device,
            steepness=10.0,
            distribution="cauchy",
        )

    def _load_model(self, ckpt: str):
        if Path(ckpt).is_dir():
            return ExcessPhysicsModel.from_pretrained(ckpt)
        return ExcessPhysicsLightningModel.load_from_checkpoint(ckpt).model

    @property
    def inventory(self):
        inventory = pd.read_csv(
            Path(__file__).absolute().parent.parent.parent.parent.joinpath("data", "mixtures", "excess_v5.csv")
        )
        return list(set(np.append(inventory.smi1.unique(), inventory.smi2.unique())))

    @property
    def inventory_matrix(self):

        enc = self.tokenizer(
            self.inventory,
            padding=True, 
            truncation=True, 
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)  # (n, L_max)
        attention_mask = enc["attention_mask"].to(device)  # (n, L_max)

        with torch.no_grad():
            hs = self.model.encoder(
                input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state  # (n, L_max, E)
            E = masked_mean_pool(hs, attention_mask)  # (n, E)
        return E

    def cost(self, x, c_logits):
        # soft top‑2 selection
        _, S = self.sorter(x.unsqueeze(0))  # S: (1, n, 2)
        S = S.squeeze(0)  # (n, 2)

        # mixture embeddings from E
        mixture_embeds = S.T @ self.inventory_matrix  # (2, D)
        embs = mixture_embeds.unsqueeze(0)  # (1, 2, D)

        # soft composition over those two
        comp = F.softmax(c_logits, dim=0).view(1, 2)  # (1,2)

        y_rel = self.forward(comp, embs)
        mx, _ = torch.max(torch.abs(y_rel), dim=1)
        return -mx.sum()

    def forward(self, comp, embs):
        model = self.model
        y, _, y_excess = model.compute_interactions(comp, embs, torch.tensor(self.temperature))
        # loss = -∑_t max |y_excessₜ / yₜ|
        y_rel = y_excess / y
        return y_rel

    def run_optimization(self):
        x = torch.randn(len(self.inventory), device=device, requires_grad=True)  # inventory scores
        c_logits = torch.randn(2, device=device, requires_grad=True)  # raw mix logits
        optimizer = LBFGS([x, c_logits], lr=1.0, max_iter=20)
     

        def closure():
            optimizer.zero_grad()
            loss = self.cost(x, c_logits)
            loss.backward()
            print(
                f"loss={loss.item():.4f}, ‖x.grad‖={x.grad.norm().item():.4f}, ‖c.grad‖={c_logits.grad.norm().item():.4f}"
            )
            return loss

        optimizer.step(closure)

        _, top2 = torch.topk(x, k=2)
        best_comps = [self.inventory[i] for i in top2]
        best_fracs = F.softmax(c_logits, dim=0).tolist()

        print("Selected compounds:", best_comps)
        print("Mixture fractions :", best_fracs)


if __name__ == "__main__":
    name_or_path = "/Users/anoushka/VSCodeProjects/electrolyte-fm/data/models/mist-mixtures-17o0xho9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = ExcessOptimizer(name_or_path, temperature = 298.15)
    opt.run_optimization()