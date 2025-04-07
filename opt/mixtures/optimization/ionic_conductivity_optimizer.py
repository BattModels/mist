from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import DataCollatorWithPadding
from typing import Dict, List, Optional

from electrolyte_fm.models.model_utils import DeepSpeedMixin


class MixtureMolecularOptimization:
    """
    Optimizes a binary mixture with respect to
    a molecular substance present in the mixture.

    Parameters
    ----------
    pretrained_ckpt: str
        Path to pre-trained MixtureModelWithPhysics.
    substance_1: str
        SMILES for first substance
        (fixed during optimization).
    x1: float
        Mole fraction for substance 2
        (mixture is optimized at this compositon).
    substance_2: str
        SMILES for second substance
        (optimization design variable).
    x2: float
        Mole fraction for substance 2.
    temperature: float
        Temperature to optimize mixture property
        (in Kelvin).
    """

    def __init__(
        self, pretrained_ckpt: str, temperature: float, inventory: Optional[str] = None
    ) -> None:
        self.pretrained_ckpt = pretrained_ckpt
        self._load_model()
        self.data_collator = DataCollatorWithPadding(self.model.tokenizer, "longest")
        self.temperature = temperature
        if inventory:
            self.inventory = np.load(inventory).astype("float64")
        self.norm = torch.nn.Softmax(dim=0)

    def _load_model(self):
        if Path(self.pretrained_ckpt).exists():
            self.model = DeepSpeedMixin.load(self.pretrained_ckpt)
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self.pretrained_ckpt,
                trust_remote_code=True,
            )
        self.model.eval()

    def compute_embeddings(self, smiles: List[str]) -> Dict[str, np.ndarray]:
        """
        Creates a dictionary with SMILES as keys and embeddings as values.
        """
        batch = self.model.tokenizer(smiles)
        batch = self.data_collator(batch)
        embedding = self.model.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).last_hidden_state.mean(axis=1)
        return embedding.detach()

    def run_optimization(
        self,
        target_prop_value: float = 1.4,
        max_iter: int = 100,
        abs_ftol: int = 1e-5,
        lr: float = 3e-5,
    ):
        max_iter = int(max_iter)
        smiles = [
            "COC(=O)OC",
        ]
        batch = self.model.tokenizer(smiles)
        batch = self.data_collator(batch)
        initial_guess_ek = (
            self.model.encoder(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            .last_hidden_state.mean(axis=1)
            .detach()
        )
        initial_guess_x = torch.randn(self.inventory.shape[0])

        # Design variables
        e_k = torch.nn.Parameter(initial_guess_ek)
        x_comp = torch.nn.Parameter(initial_guess_x)

        # Define the optimizer
        optimizer = torch.optim.LBFGS(
            [e_k, x_comp],
            max_iter=max_iter,
            lr=lr,  # line_search_fn="strong_wolfe"
        )
        inv = torch.from_numpy(self.inventory.T).float()

        # Track the optimizer trajectory
        trajectory = []

        def closure():
            optimizer.zero_grad()
            lg_k = self.model.task_network(e_k, torch.tensor(self.temperature))
            x = self.norm(x_comp)
            x[14] = 0.1
            salt_comp_loss = (x[14] - x[12] - x[13]) ** 2
            # x_norm_loss = (1.0 - x.sum())**2
            sparsity_loss = 1e-5 * torch.sum(x > 1e-16)
            inventory_loss = ((torch.matmul(inv, x) - e_k) ** 2).sum()
            exploitation_loss = (lg_k - target_prop_value) ** 2

            print(f"sparsity_loss {sparsity_loss} inventory_loss {inventory_loss}")
            print(
                f"exploitation_loss {exploitation_loss} salt_comp_loss {salt_comp_loss}"
            )

            loss = (
                5 * exploitation_loss
                + inventory_loss
                + 7 * salt_comp_loss
                + sparsity_loss
            )
            loss.backward(retain_graph=True)
            return loss

        # Perform optimization
        for it in range(max_iter):
            loss = optimizer.step(closure)
            P = self.model.task_network(e_k, torch.tensor(self.temperature))
            trajectory.append(float(P.detach()[0]))
            print(f"iteration {it} P {P.detach()[0]} loss {loss.detach()[0]}")
            if it > 5 and loss < abs_ftol:
                print(f"P {P} target_prop_value {target_prop_value}")
                break
        print(f"iterations required {it}")
        return e_k.detach(), self.norm(x_comp).detach(), trajectory

    def save_inventory(self, inventory_size, run_id):
        df = pd.read_csv(
            "/home/abhutani/electrolyte-fm/diffmix_data/Ion_Cond_Tle20.csv"
        )
        dfi = pd.read_csv("inventory.csv")
        smiles = []
        for col in [f"smi{i}" for i in range(1, 6)]:
            smiles.extend(df[col].unique())
        smiles.extend(dfi.smiles)
        df = pd.DataFrame({"smiles": smiles[:inventory_size]})
        df.to_csv(f"inventory_smiles_{run_id}_{inventory_size}.csv")
        unique_smiles = list(set(smiles))
        inventory_collector = []
        step = 2000
        for idx in range(0, inventory_size, step):
            inv = self.compute_embeddings(unique_smiles[idx : idx + step]).numpy()
            inventory_collector.append(inv)
        inventory = np.concatenate(inventory_collector)
        print(inventory.shape)
        with open(f"inventory_{run_id}_{inventory_size}.npy", "wb") as f:
            np.save(f, inventory)
        self.inventory = inventory.astype("float64")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import niceplots

    # run_id = "z4ni8hcj"
    run_id = "c7ssptg7"
    pretrained_ckpt = (
        f"/home/abhutani/electrolyte-fm/mist/{run_id}/checkpoints/last.ckpt"
    )

    inventory_size = 4000

    temp = 294
    inventory = f"inventory_{run_id}_{inventory_size}.npy"
    opt = MixtureMolecularOptimization(pretrained_ckpt, temp, inventory=None)
    opt.save_inventory(
        inventory_size,
        run_id,
    )

    target_prop_value = 2.6
    e_k, x, trajectory = opt.run_optimization(
        max_iter=800, target_prop_value=target_prop_value, lr=4e-3
    )
    smiles = pd.read_csv(
        f"inventory_smiles_{run_id}_{inventory_size}.csv"
    ).smiles.values
    results = pd.DataFrame({"smiles": smiles[:inventory_size], "x": x.numpy()})
    results.sort_values(by="x", inplace=True)
    results.to_csv(f"results_{run_id}_{inventory_size}.csv")
    pd.DataFrame({"trajectory": trajectory}).to_csv(
        f"trajectory_{run_id}_{inventory_size}.csv"
    )

    with open(f"optimized_emb_{run_id}_{inventory_size}.npy", "wb") as f:
        np.save(f, e_k.numpy())

    fig, ax = plt.subplots()
    plt.style.use(niceplots.get_style())
    colors = niceplots.get_colors()
    ax.axhline(y=np.exp(target_prop_value), c=colors["Orange"])
    trajectory = [np.exp(i) for i in trajectory[::30]]
    (line,) = ax.plot(
        trajectory,
        "-o",
        markeredgecolor="w",
        linewidth=2.0,
        markersize=8,
        clip_on=False,
    )
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Ionic Conductivity [mS/cm]", rotation="horizontal", ha="right")
    niceplots.adjust_spines(ax)
    niceplots.save_figs(fig, f"ionic_cond_traj_{run_id}_{inventory_size}", ["pdf"])
