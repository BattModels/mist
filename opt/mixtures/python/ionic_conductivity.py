import itertools
import torch
from typing import Optional
from accelerate import Accelerator
from pathlib import Path

from smirk import SmirkTokenizerFast

from transformers import DataCollatorWithPadding
from torch.utils.data import IterableDataset
from electrolyte_fm.models.prod_mixture import MISTIonicConductivity


def generate_simplex_grid(n, grid_size, fixed_last=None):
    g = grid_size

    if fixed_last is None:
        # full simplex
        for counts in itertools.product(range(g + 1), repeat=n - 1):
            s = sum(counts)
            if s <= g:
                last = g - s
                yield [*(c / g for c in counts), last / g]
    else:
        # Fix last coordinate
        if isinstance(fixed_last, int):
            k = fixed_last
        else:
            # float → nearest grid step
            k = int(round(float(fixed_last) * g))
        if not (0 <= k <= g):
            raise ValueError(f"fixed_last={fixed_last} maps to k={k}, expected 0..{g}")

        target_sum = g - k

        # Degenerate slice: salt = 1.0 -> only one point
        if target_sum == 0:
            yield [*(0.0 for _ in range(n - 1)), k / g]
            return

        # edge vertices first
        # one solvent takes all remaining fraction, others 0
        for i in range(n - 1):
            counts = [0] * (n - 1)
            counts[i] = target_sum
            yield [*(c / g for c in counts), k / g]

        # Enumerate only solvent counts summing to target_sum
        # (n-1 solvents; last coord is fixed to k/g)
        for counts in itertools.product(range(target_sum + 1), repeat=n - 1):
            if sum(counts) == target_sum:
                yield [*(c / g for c in counts), k / g]


def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj


class MixtureDataset(IterableDataset):
    def __init__(
        self,
        mixtures: list[dict],
        tokenizer=None,
        n: int = 50,
        fixed_salt: Optional[float] = None,
    ):
        self.tokenizer = tokenizer or SmirkTokenizerFast()
        self.token_collator = DataCollatorWithPadding(self.tokenizer)
        self.mixtures = mixtures
        self.n = n
        self.fixed_salt = fixed_salt

    def __iter__(self):
        n_components = 5
        for mixture in self.mixtures:
            components = list(mixture["solvents"])
            salt = mixture["salt"]
            assert len(salt) == 2, "Specify cation and anion SMILES"
            components.extend(salt)

            temperature = mixture.get("temperature", 293.15)

            assert (
                len(components) == n_components
            ), "Number of components must be consistent"
            obs = {
                "components": components,
                "temperature": torch.tensor(temperature),
            }

            # Anion and cation have the same mole ratio
            for comp in generate_simplex_grid(
                n_components - 1, self.n, fixed_last=self.fixed_salt
            ):
                yield {**obs, "composition": torch.tensor(comp)}

    def get_dataloader(self, batch_size=16):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        out = {
            "temperature": torch.tensor(
                [x["temperature"] for x in batch], dtype=torch.float32
            )
        }

        # Get number of components from first sample
        n_components = len(batch[0]["components"])

        for idx in range(n_components):
            # Collect SMILES for this component across all samples in batch
            smiles_list = [x["components"][idx] for x in batch]

            # Collect composition for this component across all samples
            if idx == n_components - 1:
                # salt molarity:
                composition = [x["composition"][-1] for x in batch]
            else:
                composition = [x["composition"][idx] for x in batch]

            # Tokenize all SMILES for this component
            enc = self.tokenizer(smiles_list)
            enc = self.token_collator(enc)

            out[f"input_ids_{idx}"] = enc["input_ids"]
            out[f"attention_mask_{idx}"] = enc["attention_mask"]
            out[f"composition_{idx}"] = torch.tensor(composition, dtype=torch.float32)

        out["components"] = [x["components"] for x in batch]
        out["composition"] = [x["composition"] for x in batch]
        # Add target if present
        if "target" in batch[0]:
            out["target"] = torch.tensor(
                [x["target"] for x in batch], dtype=torch.float32
            )

        return out


def evaluate_mixtures(model, mixtures: list[dict], n=15, fixed_salt=None, **kwargs):
    ds = MixtureDataset(mixtures, n=n, fixed_salt=fixed_salt)
    dl = ds.get_dataloader()
    return evaluate(model, dl, **kwargs)


def evaluate(model, dataloader):
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare_model(model.eval(), device_placement=True)

    for batch in dataloader:
        batch = move_to_device(batch, device)
        comps = torch.stack([c for c in batch["composition"]])
        assert torch.isfinite(comps).all(), "Non-finite in batch composition."
        assert (comps >= -1e-6).all() and (
            comps <= 1 + 1e-6
        ).all(), "Composition out of [0,1]."
        assert torch.allclose(
            comps.sum(-1), torch.ones(comps.shape[0]), atol=1e-6
        ), "Row sum != 1."

        output = model.forward(batch, return_all=True)[-1]
        targets = list(output.keys())
        for bdx in range(batch["input_ids_0"].shape[0]):
            out = {}
            out["components"] = batch["components"][bdx]
            out["composition"] = batch["composition"][bdx].tolist()
            out["temperature"] = batch["temperature"][bdx].item()
            for target in targets:
                out[target] = output[target][bdx].tolist()
            yield out


def load_conductivity_model(ckpt):
    return MISTIonicConductivity.from_pretrained(ckpt)


def evaluate_at_composition(model, mixture, composition):
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare_model(model.eval(), device_placement=True)

    tokenizer = SmirkTokenizerFast()
    collator = DataCollatorWithPadding(tokenizer)

    solvents = list(mixture["solvents"])
    salt = list(mixture["salt"])
    assert len(solvents) == 3, "Expected three solvents."
    assert len(salt) == 2, "Specify [anion, cation] SMILES."
    components = solvents + salt  # [s1, s2, s3, anion, cation]
    temperature = float(mixture.get("temperature", 293.15))

    comp = torch.tensor(list(composition), dtype=torch.float32)
    assert comp.numel() == 4, "composition must be length 4: [s1, s2, s3, salt]"
    comp = torch.clamp(comp, 0.0, 1.0)
    comp[-1] = 1.0 - comp[:-1].sum()  # enforce exact sum==1.0 in fp32

    batch = {"temperature": torch.tensor([temperature], dtype=torch.float32)}
    for idx in range(5):
        smiles = [components[idx]]
        enc = tokenizer(smiles)
        enc = collator(enc)
        batch[f"input_ids_{idx}"] = enc["input_ids"]
        batch[f"attention_mask_{idx}"] = enc["attention_mask"]
        if idx < 4:
            batch[f"composition_{idx}"] = comp[idx].reshape(1)
        else:
            batch[f"composition_{idx}"] = comp[-1].reshape(1)

    batch["components"] = [components]
    batch["composition"] = [comp]
    batch = move_to_device(batch, device)

    with torch.no_grad():
        outputs = model.forward(batch, return_all=True)[-1]
    targets = list(outputs.keys())

    result = {
        "components": components,
        "composition": comp.tolist(),
        "temperature": temperature,
    }
    for t in targets:
        result[t] = outputs[t][0].detach().cpu().tolist()
    return result


if __name__ == "__main__":
    name_or_path = "/Users/anoushka/VSCodeProjects/electrolyte-fm/data/models/mist-conductivity-27.0M-2mpg8dcd/"
    model = MISTIonicConductivity.from_pretrained(name_or_path)
    mixtures = [
        {
            "solvents": ["O=C1OCC(F)O1", "CCOC(=O)OC", "O=C1OCCO1"],
            "salt": ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "[Li+]"],
            "temperature": 293.15,
        },
    ]
    # print(evaluate_at_composition(model, mixtures[0], [0.9, 0, 0, 0.1]))
    for out in evaluate_mixtures(model, mixtures, n=5, fixed_salt=0.1):
        print(out["composition"])
