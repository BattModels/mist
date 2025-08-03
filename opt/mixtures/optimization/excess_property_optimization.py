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
from excess import evaluate_mixtures, generate_simplex_grid
from utils import process_prediction_with_ref

@dataclass
class OptimizationConfig:
    lr: float
    num_iterations: int
    target_quantities: list[str]

class ExcessOptimizer:
    def __init__(
        self, pretrained_ckpt: str, temperature: float, config: OptimizationConfig
    ) -> None:
        self.pretrained_ckpt = pretrained_ckpt
        self.model = self._load_model(self.pretrained_ckpt)
        self.tokenizer = SmirkTokenizerFast()
        self.temperature = temperature
        self.n = 5
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

    def cost(self, x: torch.Tensor):
        # soft top‑2 selection
        _, S = self.sorter(x.unsqueeze(0))  # S: (1, n, 2)
        S = S.squeeze(0)  # (n, 2)

        # mixture embeddings from E
        mixture_embeds = S.T @ self.inventory_matrix  # (2, D)
        embs = mixture_embeds.unsqueeze(0).repeat(self.n, 1, 1)  # (N, 2, D)
        comp = torch.linspace(0, 1, steps = self.n)
        comp = torch.vstack((comp, 1 - comp)).T
        y_rel = self.forward(comp, embs)
        mask = torch.tensor([i in self.config.target_quantities for i in self.model.config.target_columns]).unsqueeze(0).repeat(self.n, 1)
        y_masked = torch.masked_select(y_rel, mask)        
        if y_masked.ndim > 1:
            mx, _ = torch.max(torch.abs(y_masked), dim=1)
        else:
            mx = torch.abs(y_masked).max()
        self.target_trajectory.append(mx.detach())
        return -mx.sum()

    def forward(self, comp: torch.Tensor, embs: torch.Tensor):
        model = self.model
        temperature = torch.tensor(self.temperature).repeat(self.n, 1, 1)
        y, _, y_excess = model.compute_interactions(comp, embs, temperature)
        # loss = -∑_t max |y_excessₜ / yₜ|
        y_rel = y_excess / y
        return y_rel

    def run_optimization(self):
        x = torch.randn(len(self.inventory), device=device, requires_grad=True)  # inventory scores
        optimizer = LBFGS([x], lr=self.config.lr, max_iter=self.config.num_iterations)
     
        def closure():
            optimizer.zero_grad()
            loss = self.cost(x)
            loss.backward()
            print(
                f"loss={loss.item():.4f}, ‖x.grad‖={x.grad.norm().item():.4f}"
            )
            return loss

        optimizer.step(closure)

        _, top2 = torch.topk(x, k=2)
        self.optimal_mixture = [
            {
                "compounds": [self.inventory[i] for i in top2],
                "temperature": self.temperature
            },
        ]
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
        n_targets = len(self.config.target_quantities)
        for idx, target in enumerate(self.config.target_quantities):
            if n_targets > 1:
                ax.plot(self.target_trajectory[:, idx], label = target)
            else:
                ax.plot(self.target_trajectory[:], label = target)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("% Excess")
        fig.legend()
        return fig

    
    def plot_optimization_results(self, results_df: pd.DataFrame):
        fig, ax = plt.subplots()
        x1 = [i[0] for i in results_df.composition]
        for target in self.config.target_quantities:
            y_rel = results_df[f"relative excess {target}"]
            ax.plot(x1, y_rel, label = target)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("% Excess")
        fig.legend()
        return fig


        
        


if __name__ == "__main__":
    
    name_or_path = "/Users/anoushka/VSCodeProjects/electrolyte-fm/data/models/mist-mixtures-17o0xho9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OptimizationConfig(lr=1, num_iterations=30, target_quantities = ["density [gram / centimeter ** 3]"])
    opt = ExcessOptimizer(name_or_path, temperature = 298.15, config=config)
    opt.run_optimization()
    opt.save_run()