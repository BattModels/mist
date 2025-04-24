import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.func import jacrev, hessian, vmap
from torchmin import minimize
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List, Optional
from scipy.spatial.distance import cosine
from pytorch_lightning import LightningModule

from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.data_modules.mixture_dataset import (
    collate_components_and_environment,
)


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
        self,
        pretrained_ckpt: str,
        substance_1: str,
        x1: float,
        substance_2: str,
        x2: float,
        temperature: float,
    ) -> None:
        self.pretrained_ckpt = pretrained_ckpt
        self.n_components = 2  # Only supports binary mixtures
        self._load_model()
        self.data_collator = DataCollatorWithPadding(self.model.tokenizer, "longest")
        self.substance_1 = substance_1
        self.x1 = x1
        self.substance_2 = substance_2
        self.x2 = x2
        self.temperature = temperature
        self._get_initial_batch()

    def collator(self, batch):
        output = {}
        for i in range(2):
            output[f"input_ids_{i}"] = torch.stack(
                [
                    torch.tensor(batch[f"input_ids_{i}"], dtype=int),
                ]
            )
            output[f"attention_mask_{i}"] = torch.stack(
                [
                    torch.tensor(batch[f"attention_mask_{i}"], dtype=int),
                ]
            )
            output[f"composition_{i}"] = torch.stack(
                [
                    torch.tensor(batch[f"composition_{i}"], dtype=torch.float32),
                ]
            )
        output["temperature"] = torch.stack(
            [
                torch.tensor(batch["temperature"], dtype=torch.float32),
            ]
        )
        return output

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

    def _get_initial_batch(self):
        target = 0.0  # Dummy value to initialize batch
        batch = collate_components_and_environment(
            self.substance_1,
            self.x1,
            self.substance_2,
            self.x2,
            self.temperature,
            target,
            include_temperature=True,
            tokenizer=self.model.tokenizer,
            n_components=self.n_components,
            collate=self.data_collator,
        )
        self.initial_batch = self.collator(batch)

    def batch_with_embeddings(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Replaces tokenizer representation of SMILES
        (input_ids and attention_masks) in batch with
        embedding from encoder.
        """
        temperature = batch["temperature"]
        for i in range(self.model.n_components):
            embedding = self.model.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state.mean(axis=1)
            embedding = torch.hstack((temperature.view(-1, 1), embedding))
            batch[f"embedding_{i}"] = embedding.float()
        return batch

    def compute_embeddings(self, smiles: str) -> Dict[str, np.ndarray]:
        """
        Creates a dictionary with SMILES as keys and embeddings as values.
        """
        batch = self.model.tokenizer(smiles.tolist())
        batch = self.data_collator(batch)
        embedding = self.model.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).last_hidden_state.mean(axis=1)
        return dict(zip(smiles, embedding.detach().numpy().tolist()))

    def embedding_to_nearest_smiles(
        self, embedding: torch.Tensor, all_smiles: List[str]
    ):
        """
        Convert embedding to SMILES by finding the nearest neighbour
        using Manhattan distance in embedding space.
        """
        embedding_lookup = self.compute_embeddings(all_smiles)
        distances = {
            cosine(embedding.detach().numpy()[:-1], value): key
            for key, value in embedding_lookup.items()
        }
        min_key = min(distances.keys())
        return distances[min_key]

    def embedding_to_property(self, embedding, batch):
        """
        Predict mixture property using input batch with
        embeddings and compositions.
        """
        batch["embedding_1"] = embedding.float().view(1, -1)
        pred_unscaled = self.model.task_network(batch)
        return self.model.transform.forward(pred_unscaled).flatten()

    def run_optimization(
        self,
        all_smiles: List[str],
        target_prop_value: float = -0.5,
        max_iter: int = 100,
    ):
        batch = self.batch_with_embeddings(self.initial_batch)
        e_k = batch["embedding_1"].flatten()

        # Design variable
        e_k = torch.nn.Parameter(e_k)

        # Define the optimizer
        optimizer = torch.optim.LBFGS(
            [e_k], max_iter=max_iter, lr=1e-6, line_search_fn="strong_wolfe"
        )

        # Track the optimizer trajectory
        trajectory = [
            e_k,
        ]

        def closure():
            optimizer.zero_grad()
            P = self.embedding_to_property(e_k, batch)
            loss = P - target_prop_value
            loss.backward(retain_graph=True)
            return loss

        # Perform optimization
        for it in range(max_iter):
            optimizer.step(closure)
            P = self.embedding_to_property(e_k, batch)
            trajectory.append(e_k.clone().detach())
            print(f"iteration {it} P {P} ")
            if P <= target_prop_value:
                print(f"P {P} target_prop_value {target_prop_value}")
                break

        optimized_molecule = self.embedding_to_nearest_smiles(e_k.detach(), all_smiles)
        res = collate_components_and_environment(
            self.substance_1,
            self.x1,
            optimized_molecule,
            self.x2,
            self.temperature,
            0,
            include_temperature=True,
            tokenizer=self.model.tokenizer,
            n_components=self.n_components,
            collate=self.data_collator,
        )
        res = self.collator(res)
        P_final = self.model(res)
        return optimized_molecule, trajectory, P_final


# if __name__ == "__main__":
#     # TODO: move to test
#     smi1 = "CC1COC(=O)O1"  # Propylene Carbonate
#     x1 = 0.5905
#     smi2 = "C1CCOC1"  # Tetrahydrofuran
#     x2 = 0.4095
#     temperature = 298.15
#     # P_0 = 0.406
#     datapaths = glob.glob(
#         "/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_volume/*.csv"
#     )
#     df = pd.concat([pd.read_csv(fp) for fp in datapaths])
#     all_smiles = np.append(df.smi1.unique(), df.smi2.unique())
#     opt = MixtureMolecularOptimization(
#         substance_1=smi1,
#         x1=x1,
#         substance_2=smi2,
#         x2=x2,
#         temperature=temperature,
#         pretrained_ckpt="./mist/n0zaewvp/checkpoints/last.ckpt",
#     )
#     optimized_molecule, P_final = opt.run_optimization(all_smiles, max_iter=500)
#     for i in P_final:
#         print(f"e_k {i}")
