import itertools
import torch
from accelerate import Accelerator
from pathlib import Path

from smirk import SmirkTokenizerFast
from electrolyte_fm.models.excess_physics_model import (
    ExcessPhysicsLightningModel,
    ExcessPhysicsModel,
    _default_targets,
)
from electrolyte_fm.data_modules.utils import MolEncoding, collate_target
from electrolyte_fm.data_modules.mixture_dataset import (
    ComponentDataModule,
    encode_and_tokenize_mixture,
)
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import IterableDataset, default_collate


class MixtureDataset(IterableDataset):
    def __init__(self, mixtures: list[dict], tokenizer=None, n: int = 50):
        self.tokenizer = tokenizer or SmirkTokenizerFast()
        self.token_collator = DataCollatorWithPadding(self.tokenizer)
        self.mixtures = mixtures
        self.n = n

    def __iter__(self):
        n_compounds = None
        for mixture in self.mixtures:
            compounds = mixture["compounds"]
            temperature = mixture.get("temperature", 293.15)
            if n_compounds is None:
                n_compounds = len(compounds)
            assert len(compounds) == n_compounds, (
                "Number of compounds must be consistent"
            )
            obs = {
                "compounds": compounds,
                "temperature": torch.tensor(temperature),
            }

            for comp in generate_simplex_grid(n_compounds, self.n):
                yield {**obs, "composition": torch.tensor(comp)}

    def get_dataloader(self, batch_size=16):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        out = {
            k: default_collate([obs[k] for obs in batch])
            for k in batch[0].keys()
            if k not in ["compounds"]
        }
        out.update(
            encode_and_tokenize_mixture(
                [obs["compounds"] for obs in batch],
                tokenizer=self.tokenizer,
                encoding=MolEncoding.KEKULE,
                randomize=False,
                token_collator=self.token_collator,
                include_encoding=True,
            )
        )
        return out


class MixtureCSVDataset(MixtureDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer=None,
        target_columns: list[str] | None = None,
        excess_columns: str = "excess {:s}",
    ):
        self.tokenizer = tokenizer or SmirkTokenizerFast()
        self.token_collator = DataCollatorWithPadding(self.tokenizer)
        self.target_columns = target_columns or _default_targets()
        self.excess_columns = [
            excess_columns.format(col) for col in self.target_columns
        ]
        self.dataset = load_dataset(
            "csv",
            data_files=str(path),
            keep_in_memory=False,
            split="train",
            streaming=True,
        )

    @classmethod
    def format_row(cls, row):
        obs = {
            "compounds": [row["smi1"], row["smi2"]],
            "temperature": torch.tensor(
                row["temperature [kelvin]"], dtype=torch.float32
            ),
            "composition": torch.tensor(
                [row["x1"], 1 - row["x1"]], dtype=torch.float32
            ),
        }
        return obs

    def __iter__(self):
        yield from (
            self.dataset.map(self.format_row)
            .map(
                collate_target,
                batched=False,
                fn_kwargs={
                    "target_columns": self.target_columns,
                    "dtype": torch.float32,
                },
            )
            .map(
                collate_target,
                batched=False,
                fn_kwargs={
                    "target_columns": self.excess_columns,
                    "name": "target_excess",
                    "dtype": torch.float32,
                },
            )
            .select_columns(
                [
                    "compounds",
                    "temperature",
                    "composition",
                    "target",
                    "target_mask",
                    "target_excess",
                    "target_excess_mask",
                ]
            )
            .with_format("torch")
        )


def generate_simplex_grid(n, grid_size):
    """
    Generate a grid of evenly spaced points in an n-simplex using PyTorch.

    Parameters:
    - n: the dimension of the simplex (e.g., for a triangle, n=2, for a tetrahedron, n=3)
    - grid_size: number of divisions along each axis (number of points along one edge)

    Returns:
    - A tensor of shape (num_points, n+1), where each row represents a point
      in the n-simplex using barycentric coordinates.
    """
    grid_values = (i / grid_size for i in range(grid_size + 1))

    # Generate all combinations of grid values for the n-1 dimensions
    grid_combinations = itertools.product(grid_values, repeat=n - 1)

    # For each combination, scale it such that the sum of the coordinates is 1
    for combination in grid_combinations:
        # Calculate the last coordinate to make sure the sum is 1
        last_coord = 1 - sum(combination)

        # Ensure that the last coordinate is non-negative (valid point inside the simplex)
        if last_coord >= 0:
            # Yield the full point with the last coordinate
            yield [*combination, last_coord]


def load_excess_model(ckpt):
    if Path(ckpt).is_dir():
        return ExcessPhysicsModel.from_pretrained(ckpt)
    return ExcessPhysicsLightningModel.load_from_checkpoint(ckpt).model


def evaluate_mixtures(model: ExcessPhysicsModel, mixtures: list[dict], n=50, **kwargs):
    ds = MixtureDataset(mixtures, n=n)
    dl = ds.get_dataloader()
    return evaluate(model, dl, **kwargs)


def evaluate_binary_csv(model: ExcessPhysicsModel, path: str | Path, **kwargs):
    ds = MixtureCSVDataset(path)
    return evaluate(model, ds.get_dataloader(), **kwargs)


def evaluate_dataset(model: ExcessPhysicsModel, path, **kwargs):
    assert Path(path).exists()
    dm = ComponentDataModule(
        path,
        batch_size=16,
        target_columns=model.config.target_columns,
        include_encoding=True,
    )
    dm.prepare_data()
    dm.setup("fit")

    return evaluate(model, dm.val_dataloader(), **kwargs)


@torch.no_grad()
def grad_target(y, target_idx):
    yg = torch.zeros_like(y)
    yg[:, target_idx] = 1
    return yg


def compute_gradients(y, **xs):
    grads = {k: [] for k in xs.keys()}
    for tdx in range(y.shape[1]):
        yg = grad_target(y, tdx)
        yg.requires_grad = True
        y.backward(yg, retain_graph=True)
        for k, x in xs.items():
            grads[k].append(x.grad.detach().clone())
            x.grad = None

    grads = {k: torch.stack(grad, dim=1) for k, grad in grads.items()}
    return grads


def evaluate(model: ExcessPhysicsModel, dataloader, gradients: bool = False):
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare_model(model.eval(), device_placement=True)

    for batch in dataloader:
        smi_columns = [
            k for k in batch.keys() if not isinstance(batch[k][0], torch.Tensor)
        ]
        temperature = batch["temperature"].to(device)
        composition = batch["composition"].to(device)
        temperature.requires_grad = True
        composition.requires_grad = True
        y, y_linear, y_excess = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            temperature=temperature,
            composition=composition,
        )

        # Compute gradients
        y_grads = None
        y_excess_grads = None
        if gradients:
            y_grads = compute_gradients(y, T=temperature, x=composition)
            y_excess_grads = compute_gradients(y_excess, T=temperature, x=composition)

        for bdx in range(batch["input_ids"].shape[0]):
            out = {
                **{k: batch[k][bdx] for k in smi_columns},
                "temperature": batch["temperature"][bdx].item(),
                "composition": batch["composition"][bdx].tolist(),
                "y": y[bdx].tolist(),
                "y_excess": y_excess[bdx].tolist(),
                "y_linear": y_linear[bdx].tolist(),
            }
            if gradients:
                out["dy_dT"] = y_grads["T"][bdx, :].cpu()
                out["dy_excess_dT"] = y_excess_grads["T"][bdx, :].cpu()
                out["dy_dx"] = y_grads["x"][bdx, :].cpu()
                out["dy_excess_dx"] = y_excess_grads["x"][bdx, :].cpu()

            if "target" in batch:
                out["target"] = batch["target"][bdx].tolist()
                out["target_mask"] = batch["target_mask"][bdx].tolist()
            if "target_excess" in batch:
                out["target_excess"] = batch["target_excess"][bdx].tolist()
                out["target_excess_mask"] = batch["target_excess_mask"][bdx].tolist()

            yield out


if __name__ == "__main__":
    dataset = "/Users/alexwadell/electrolyte-fm/mixtures/excess_dataset_v5/random"
    name_or_path = (
        "~/Downloads/z8ido8hw/checkpoints/epoch=61-step=2232-val_loss=1.232.ckpt"
    )
    model = load_excess_model(name_or_path)
    for out in evaluate_binary_csv(
        model,
        "/Users/alexwadell/Documents/repos/excess_density/excess_v5.csv",
        gradients=True,
    ):
        print(out)

    mixtures = [
        {"compounds": ["CC#N", "CCCO"], "temperature": 293.15},
        {"compounds": ["CC#N", "CO"], "temperature": 293.15},
        {"compounds": ["CC#N", "CCCCCCCCCCO"], "temperature": 293.15},
    ]
    for out in evaluate_mixtures(model, mixtures, gradients=True):
        print(out)
