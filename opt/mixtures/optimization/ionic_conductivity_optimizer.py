from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from pyoptsparse import OPT, History, Optimization
from rdkit import Chem
from transformers import DataCollatorWithPadding

from electrolyte_fm.models.model_utils import DeepSpeedMixin

torch.autograd.set_detect_anomaly(True)


def encode(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


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
        self.data_collator = DataCollatorWithPadding(
            self.model.tokenizer, "longest", return_tensors="pt"
        )
        self.temperature = temperature
        self.constraint = []
        self.trajectory = []
        if inventory:
            self.inventory = np.load(inventory).astype("float64")
        self.norm = lambda x: x  # torch.nn.Softmax(dim=0)

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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).last_hidden_state.mean(axis=1)
        return embedding.detach()

    def save_inventory(self, run_id):
        df = pd.read_csv(
            "/home/abhutani/electrolyte-fm/diffmix_data/Ion_Cond_Tle20.csv"
        )
        # dfi = pd.read_csv("inventory.csv")
        smiles = [
            "CCOC(=O)OC",
            "CCOC(=O)OC(F)(F)F",
            "COC(=O)OCC(F)(F)F",
            "CCOC(=O)OCC(F)(F)F",
            "C1CCC(CC1)F",
            "C1=CC(=C(C=C1F)F)F",
            "C(C(F)(F)F)OC(=O)OCC(F)(F)F",
            "C(COCC(F)(F)F)OCC(F)(F)F",
            "C1(C(OC(=O)O1)(F)F)F",
            "C(C(C(F)F)(F)F)OC(=O)OCC(C(F)F)(F)F",
        ]
        for col in [f"smi{i}" for i in range(1, 6)]:
            smiles.extend(df[col].unique())
        smiles = [encode(smi) for smi in smiles]
        unique_smiles = list(set(smiles))
        df = pd.DataFrame({"smiles": unique_smiles})
        self.inventory_size = len(unique_smiles)
        df.to_csv(f"inventory_smiles_{run_id}_{self.inventory_size}.csv")
        inventory_collector = []
        # for idx in range(0, self.inventory_size, 2):
        inv = self.compute_embeddings(unique_smiles).numpy()
        inventory_collector.append(inv)
        inventory = np.concatenate(inventory_collector)
        print(inventory.shape)
        with open(f"inventory_{run_id}_{self.inventory_size}.npy", "wb") as f:
            np.save(f, inventory)
        self.inventory = inventory.astype("float64")
        self.Li = unique_smiles.index("[Li+]")
        self.PF6 = unique_smiles.index("F[P-](F)(F)(F)(F)F")
        self.TFSI = unique_smiles.index("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F")

    def run_optimization_ipopt(
        self,
        target_prop_value: float = 1.4,
        max_iter: int = 100,
        abs_ftol: float = 1e-6,
    ):
        assert hasattr(
            self, "inventory"
        ), "Inventory must be loaded before optimization."

        inv = torch.from_numpy(self.inventory.T).float()
        n_vars = self.inventory.shape[0]
        assert n_vars == self.inventory_size

        # Starting guess
        x0 = np.random.rand(n_vars)
        x0[self.Li] = x0.max()
        x0[self.TFSI] = 0.5 * x0[self.Li]
        x0[self.PF6] = 0.5 * x0[self.Li]
        x0 = np.exp(x0) / np.exp(x0).sum()

        def forward(xdict):
            x_np = xdict["x_comp"]
            x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
            x_soft = self.norm(x_tensor)
            e_k = torch.matmul(inv, x_soft).view(1, -1)
            # Constraint: cation - anions == 0
            cons_val = x_soft[self.Li] - x_soft[self.TFSI] - x_soft[self.PF6]
            lg_k, _, _, _ = self.model.task_network(e_k, torch.tensor(self.temperature))

            exploitation_loss = (target_prop_value - lg_k) ** 2
            # Gradient w.r.t objective
            exploitation_loss.backward()
            self.trajectory.append(np.exp(lg_k.detach()))
            return x_tensor, cons_val, exploitation_loss

        def objective_function(xdict):
            x_tensor, cons_val, exploitation_loss = forward(xdict)

            self.constraint.append(cons_val)
            funcs = {
                "obj": float(exploitation_loss.detach().item()),
                "salt_composition": float(cons_val),
                "unity": float(x_tensor.detach().sum() - 1),
            }

            fail = False
            return funcs, fail

        def sensitivity_function(xdict, funcsDict):
            x_tensor, cons_val, exploitation_loss = forward(xdict)
            grad_obj = x_tensor.grad.detach().numpy()

            # Gradient w.r.t. x is zero except at ion conc index
            grad_cons = np.zeros_like(x_tensor.detach(), dtype=float)
            grad_cons[self.Li] = 1
            grad_cons[self.TFSI] = -1
            grad_cons[self.PF6] = -1
            grads = {
                "obj": {"x_comp": grad_obj},
                "salt_composition": {"x_comp": grad_cons},
                "unity": {"x_comp": np.ones_like(x_tensor.detach(), dtype=float)},
            }
            fail = False
            return grads, fail

        # Optimization problem defintion
        optProb = Optimization("MixtureOpt", objective_function)
        optProb.addVarGroup(
            "x_comp",
            n_vars,
            value=x0,
            lower=0.0,
            upper=1.0,
        )
        optProb.addObj("obj")
        optProb.addCon("salt_composition", lower=0.0, upper=0.0)
        optProb.addCon("unity", lower=0.0, upper=0.0)
        # optProb.addCon("sparsity", lower=7, upper=12)

        optOptions = {
            "print_level": 5,
            "obj_scaling_factor": 5e0,
            # "nlp_scaling_obj_target_gradient": 1.5,
            "acceptable_tol": 1e-10,
            "max_iter": max_iter,
            "tol": abs_ftol,
        }
        # Initialize optimizer
        ipopt = OPT("IPOPT", options=optOptions)
        sol = ipopt(
            optProb, sens=sensitivity_function, storeHistory="trial", storeSens=True
        )

        return sol, History("trial")


if __name__ == "__main__":
    run_id = "ur7v2a6y"
    pretrained_ckpt = (
        f"/home/abhutani/electrolyte-fm/mist/{run_id}/checkpoints/last.ckpt"
    )

    opt = MixtureMolecularOptimization(pretrained_ckpt, temperature=293.15)
    opt.save_inventory(
        run_id,
    )

    solution, history = opt.run_optimization_ipopt(
        max_iter=300, abs_ftol=1e-16, target_prop_value=2.7
    )
