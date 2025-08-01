import os
import sys
import json
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime
import matplotlib.pyplot as plt
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
sys.path.append(
    Path(__file__).absolute().parent.joinpath("utils.py")
)
from excess import evaluate_mixtures
from utils import process_prediction_with_ref

@dataclass
class OptimizationConfig:
    lr: float
    num_iterations: int

class ExcessOptimizer:
    def __init__(
        self, pretrained_ckpt: str, temperature: float, config: OptimizationConfig
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
        self.config = config
        self.target_trajectory = []

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

    def cost(self, x: torch.Tensor, c_logits: torch.Tensor):
        # soft top‑2 selection
        _, S = self.sorter(x.unsqueeze(0))  # S: (1, n, 2)
        S = S.squeeze(0)  # (n, 2)

        # mixture embeddings from E
        mixture_embeds = S.T @ self.inventory_matrix  # (2, D)
        embs = mixture_embeds.unsqueeze(0)  # (1, 2, D)

        # soft composition over those two
        comp = F.softmax(c_logits, dim=0).view(1, 2)  # (1,2)

        y_rel = self.forward(comp, embs)
        self.target_trajectory.append(y_rel.detach().numpy())
        mx, _ = torch.max(torch.abs(y_rel), dim=1)
        return -mx.sum()

    def forward(self, comp: torch.Tensor, embs: torch.Tensor):
        model = self.model
        y, _, y_excess = model.compute_interactions(comp, embs, torch.tensor(self.temperature))
        # loss = -∑_t max |y_excessₜ / yₜ|
        y_rel = y_excess / y
        return y_rel

    def run_optimization(self):
        x = torch.randn(len(self.inventory), device=device, requires_grad=True)  # inventory scores
        c_logits = torch.randn(2, device=device, requires_grad=True)  # raw mix logits
        optimizer = LBFGS([x, c_logits], lr=self.config.lr, max_iter=self.config.lr)
     
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
        self.best_fracs = F.softmax(c_logits, dim=0).tolist()
        self.optimal_mixture = [
            {
                "compounds": [self.inventory[i] for i in top2],
                "temperature": self.temperature
            },
        ]
        print("Mixture fractions :", self.best_fracs)
        self.target_trajectory = np.squeeze(np.array(self.target_trajectory))
    
    def save_run(self):
        run_id = f'run_{datetime.now().strftime(format="%d%m%y_%H%M")}'
        save_path = Path(__file__).absolute().parent.joinpath("results", run_id)
        os.makedirs(save_path, exist_ok=True)
        # Save config
        with open(save_path.joinpath("config.json"), 'w') as json_file:
            json.dump(asdict(self.config), json_file, indent=4)

        # Save optimized curves
        out = evaluate_mixtures(model = self.model, mixtures = self.optimal_mixture)
        df = process_prediction_with_ref(out, targets = self.model.config.target_columns)
        df.to_csv(save_path.joinpath("optimized_curves.csv"))

        # Save trajectory
        np.savez(save_path.joinpath("trajectory.npz"), self.target_trajectory)

        # Save figures
        f_traj = self.plot_trajectory()
        f_traj.savefig(save_path.joinpath("trajectory.png"))

        f_curves = self.plot_optimization_results(df)
        f_curves.savefig(save_path.joinpath("optimal_curves.png"))

       
    
    def plot_trajectory(self):
        fig, ax = plt.subplots()
        for idx, target in enumerate(self.model.config.target_columns):
            ax.plot(self.target_trajectory[:, idx], label = target)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("% Excess")
        fig.legend()
        return fig

    
    def plot_optimization_results(self, results_df: pd.DataFrame):
        fig, ax = plt.subplots()
        x1 = [i[0] for i in results_df.composition]
        for idx, target in enumerate(self.model.config.target_columns):
            y_rel = results_df[f"relative excess {target}"]
            ax.plot(x1, y_rel, label = target)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("% Excess")
        fig.legend()
        return fig


        
        


if __name__ == "__main__":
    
    name_or_path = "/Users/anoushka/VSCodeProjects/electrolyte-fm/data/models/mist-mixtures-17o0xho9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OptimizationConfig(lr=0.1, num_iterations=30)
    opt = ExcessOptimizer(name_or_path, temperature = 298.15, config=config)
    opt.run_optimization()
    opt.save_run()