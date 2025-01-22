import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.func import jacrev, hessian, vmap
from torchmin import minimize
from transformers import DataCollatorWithPadding
from scipy.spatial.distance import cityblock


from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.data_modules.mixture_dataset import (
    ComponentDataModule,
    collate_components_and_environment,
)


def collator(x):
    output = {}
    for i in range(2):
        output[f"input_ids_{i}"] = torch.stack(
            [
                torch.tensor(x[f"input_ids_{i}"], dtype=int),
            ]
        )
        output[f"attention_mask_{i}"] = torch.stack(
            [
                torch.tensor(x[f"attention_mask_{i}"], dtype=int),
            ]
        )
        output[f"composition_{i}"] = torch.stack(
            [
                torch.tensor(x[f"composition_{i}"], dtype=torch.float32),
            ]
        )

    output["target"] = torch.stack(
        [
            torch.tensor(x["target"], dtype=torch.float32),
        ]
    )
    output["temperature"] = torch.stack(
        [
            torch.tensor(x["temperature"], dtype=torch.float32),
        ]
    )
    return output


def batch_with_embeddings(model, batch):
    temperature = batch["temperature"]
    for i in range(model.n_components):
        embedding = model.encoder(
            batch[f"input_ids_{i}"],
            attention_mask=batch[f"attention_mask_{i}"],
            return_dict=True,
            output_hidden_states=True,
        ).last_hidden_state.mean(axis=1)
        embedding = torch.hstack((temperature.view(-1, 1), embedding))
        batch[f"embedding_{i}"] = embedding.float()
    return batch


def compute_embeddings(model, smiles):
    data_collator = DataCollatorWithPadding(model.tokenizer, "longest")
    batch = model.tokenizer(smiles.tolist())
    batch = data_collator(batch)
    embedding = model.encoder(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
        output_hidden_states=True,
    ).last_hidden_state.mean(axis=1)
    return dict(zip(smiles, embedding.detach().numpy().tolist()))


def embedding_to_smiles(model, embedding):
    datapaths = glob.glob(
        "/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_volume/*.csv"
    )
    df = pd.concat([pd.read_csv(fp) for fp in datapaths])
    smiles = np.append(df.smi1.unique(), df.smi2.unique())
    embedding_lookup = compute_embeddings(model, smiles)
    distances = {
        cityblock(embedding.detach().numpy()[:-1], value): key
        for key, value in embedding_lookup.items()
    }
    min_key = min(distances.keys())  # Find the minimum key
    return distances[min_key]  # Get the value associated with the minimum key


def embedding_to_property(embedding, batch):
    batch["embedding_1"] = embedding.float().view(1, -1)
    pred_unscaled = model.task_network(batch)
    return -1 * model.transform.forward(pred_unscaled).flatten()


ckpt = (
    "/nfs/turbo/coe-venkvis/abhutani/electrolyte-fm/mist/go7uvsga/checkpoints/last.ckpt"
)
dm = ComponentDataModule(
    path="/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_volume",
    target_col="excess_molar_volume/(cm3/mol)",
    n_components=2,
    tokenizer=ckpt,
    batch_size=1,
)
dm.setup("train")

# smi1="CC1COC(=O)O1" # Propylene Carbonate
# x1=0.7764
# smi2="CCOC(=O)OCC" # Diethyl Carbonate
# x2=0.2236
# temperature=293.15
# temperature=(torch.tensor(temperature) - 273) / (400 - 273)
# batch = collate_components_and_environment(
#     smi1,
#     x1,
#     smi2,
#     x2,
#     temperature,
#     include_temperature=True,
#     tokenizer=dm.tokenizer,
#     n_components=2,
#     collate=dm.data_collator
# )

# batch = collator(batch)

# Load Model
if Path(ckpt).exists():
    model = DeepSpeedMixin.load(ckpt)
else:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        ckpt,
        trust_remote_code=True,
    )
model.eval()


for idx, batch in enumerate(dm.train_dataloader()):
    predicted_property = model.forward(batch)
    # batch = batch_with_embeddings(model, batch)
    print(predicted_property, batch["target"])
    e_k = batch["embedding_1"].flatten()
    target = -0.5
    while embedding_to_property(e_k, batch) > target:
        jacobian_f = jacrev(embedding_to_property)(e_k, batch)
        hessian_f = hessian(embedding_to_property)(e_k, batch)[0]
        step = torch.linalg.pinv(hessian_f) @ torch.transpose(jacobian_f, 0, 1)

        e_k = e_k - step.flatten()
        print(embedding_to_smiles(model, e_k))
    break
