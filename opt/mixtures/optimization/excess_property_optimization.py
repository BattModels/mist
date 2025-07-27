import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from difftopk import DiffTopkNet
from smirk import SmirkTokenizerFast
from torch.optim import LBFGS

from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsLightningModel, ExcessPhysicsModel)
from electrolyte_fm.models.model_utils import masked_mean_pool

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

        # forward‑pass from "Compute Pairwise interactions" onward
       
        y_rel = self.forward(embs, comp)
        mx, _ = torch.max(torch.abs(y_rel), dim=1)
        return -mx.sum()

    def forward(self, embs, comp):
        model = self.model
        indices = torch.triu_indices(2, 2, offset=1)
        
        B = 1
        I = 1
        D = embs.size(-1)  # embedding dimension
        temperature = torch.tensor([self.temperature], device=device).view(B, 1, 1).expand(-1, I, -1).reshape(B * I, 1)

        e_i = embs[:, indices[0]].reshape(B * I, D)
        e_j = embs[:, indices[1]].reshape(B * I, D)
        pw_coeffs = model.pairwise_interaction(e_i, e_j, temperature)  # (B*I, T, P)
        # partial concentrations
        x_t = comp[:, indices[0]] + comp[:, indices[1]]
        x_t = x_t.clamp(0, 1)
        x_i = comp[:, indices[0]] / x_t
        x_i = torch.where(x_i.abs() > 1e-8, x_i, torch.zeros_like(x_i))
        x_i = x_i.clamp(0, 1).view(B * I, 1)

        # temp‐dependence + polynomial eval
        pw_coeffs = model.temperature_dependence(pw_coeffs, temperature.view(B * I, 1, 1))
        pw = x_t.view(B * I, 1) * model.excess_polynomial(pw_coeffs, x_i)  # (B*I, T)
        y_excess = pw.reshape(B, I, -1).sum(dim=1)  # (B, T)

        # pure component prediction
        if model.config.temperature_dependence in ["concat", "locally-linear"]:
            t_e = temperature.view(B, 1, 1).expand(-1, 2, -1)
            y_target = model.component_properties(torch.cat([embs, t_e], dim=-1))
        else:
            y_target = model.component_properties(embs)
        y_target = model.temperature_dependence(y_target, temperature.view(B, 1, 1))  # (B,C,T)

        # linear mix
        y_linear = (y_target * comp.view(B, 2, 1)).sum(dim=1)  # (B, T)

        if model.config.relative_excess:
            y_excess = (1 + F.elu(y_excess)) * y_linear == y_excess / (
                y_linear + y_excess
            )

        # transforms & final
        y_linear = model.transform.forward(y_linear)
        y_excess = model.excess_transform.forward(y_excess)
        y = y_linear + y_excess  # (B, T)

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