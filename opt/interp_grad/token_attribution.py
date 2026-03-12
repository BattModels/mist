import base64
import json
import os
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import torch
from captum.attr import LayerIntegratedGradients
from plotly.subplots import make_subplots
from rdkit import Chem
from smirk import SmirkTokenizerFast
from src.plot_utils import draw_molecule_with_attributions, get_color_mapper
from transformers import AutoModel

from electrolyte_fm.models.prod_finetune import MISTFinetuned


def kekulize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    return smiles


def load_model(model_name: str, hf_hub: bool = True):
    if hf_hub:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        model = MISTFinetuned.from_pretrained(model_name)
    tokenizer = SmirkTokenizerFast()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device


def get_channels(model):
    if (
        hasattr(model.config, "config")
        and hasattr(model.config, "channels")
        and model.config.channels
    ):
        return model.config.channels
    elif hasattr(model, "channels"):
        return model.channels
    return []


def forward_fn(input_ids, attention_mask, model):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.no_grad()
def get_token_embeddings(model, input_ids):
    if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
        return model.encoder.embeddings.word_embeddings(input_ids)
    return model.get_input_embeddings()(input_ids)


def get_embedding_layer(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
        return model.encoder.embeddings.word_embeddings
    return model.get_input_embeddings()


def compute_attributions(smiles, model, tokenizer, n_steps=50, target_idx=None):
    kekule_smiles = kekulize_smiles(smiles)
    encoded = tokenizer([kekule_smiles])
    tokens = tokenizer.tokenize(kekule_smiles)

    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(encoded["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).to(device)

    pad_id = None
    if hasattr(model, "config") and hasattr(model.config, "pad_token_id"):
        pad_id = model.config.pad_token_id
    elif tokenizer is not None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = SmirkTokenizerFast().pad_token_id

    baseline_ids = torch.full_like(input_ids, pad_id)
    lig = LayerIntegratedGradients(
        lambda ids, am: forward_fn(ids, am, model),
        get_embedding_layer(model),
    )

    attr_kwargs = {
        "inputs": input_ids,
        "baselines": baseline_ids,
        "additional_forward_args": (attention_mask,),
        "return_convergence_delta": True,
        "n_steps": n_steps,
    }
    if target_idx is not None:
        attr_kwargs["target"] = target_idx

    attributions, delta = lig.attribute(**attr_kwargs)
    token_scores = attributions.sum(dim=-1) * attention_mask

    return kekule_smiles, tokens, token_scores.flatten(), delta


def get_prediction(model, tokenizer, device, kekule_smiles, target_idx=None):
    encoded = tokenizer([kekule_smiles])
    with torch.no_grad():
        input_ids = torch.tensor(encoded["input_ids"]).to(device)
        attention_mask = torch.tensor(encoded["attention_mask"]).to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(output, "logits"):
            logits = output.logits
            if (
                target_idx is not None
                and len(logits.shape) > 1
                and logits.shape[-1] > 1
            ):
                return logits[0, target_idx].item()
            else:
                return logits.squeeze(-1).item()
        else:
            pred = output[0]
            if target_idx is not None and len(pred.shape) > 0 and pred.shape[-1] > 1:
                return pred[target_idx].item()
            else:
                return pred.item()


def get_target_channel_idx(model, target_idx, channel_name):
    channels = get_channels(model)
    if channels and target_idx is None:
        if channel_name:
            target_idx = next(
                (
                    i
                    for i, ch in enumerate(channels)
                    if ch["name"].lower() == channel_name.lower()
                ),
                0,
            )
            print(f"Using channel: {channels[target_idx]['name']} (index {target_idx})")
        else:
            target_idx = 0
            print(
                f"Using default channel: {channels[target_idx]['name']} (index {target_idx})"
            )
    return target_idx


def compute_all_attributions(
    smiles_list, model, tokenizer, device, n_steps, target_idx
):
    all_scores = []
    all_tokens = []
    all_kekule_smiles = []
    predictions = {}

    for smiles in smiles_list:
        kekule_smiles, tokens, scores, delta = compute_attributions(
            smiles, model, tokenizer, n_steps, target_idx
        )
        predictions[smiles] = get_prediction(
            model, tokenizer, device, kekule_smiles, target_idx
        )
        print(f"Convergence delta for '{smiles}': {delta.item():.6f}")
        all_scores.append(scores)
        all_tokens.append(tokens)
        all_kekule_smiles.append(kekule_smiles)

    print(f"Predictions: {predictions}")
    return all_scores, all_tokens, all_kekule_smiles, predictions


def add_molecule_subplot(fig, row, kekule_smiles, tokens, scores, channel_name):
    norm, cmap = get_color_mapper(scores)
    scores_np = scores.cpu().numpy()

    y_min = scores_np.min()
    y_max = scores_np.max()
    y_padding = (y_max - y_min) * 0.15

    structure_img = draw_molecule_with_attributions(
        kekule_smiles, tokens, scores, norm, cmap
    )
    if structure_img:
        buffered = BytesIO()
        structure_img.save(buffered, format="PNG")
        fig.add_trace(
            go.Image(
                source=f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            ),
            row=row,
            col=1,
        )

    colors = [
        f"rgba({int(cmap(norm(s))[0] * 255)},{int(cmap(norm(s))[1] * 255)},{int(cmap(norm(s))[2] * 255)},{cmap(norm(s))[3]})"
        for s in scores_np
    ]

    fig.add_trace(
        go.Bar(
            x=list(range(len(tokens))),
            y=scores_np,
            text=tokens,
            textposition="outside",
            textfont_size=19,
            marker_color=colors,
        ),
        row=row,
        col=2,
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        row=row,
        col=1,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        row=row,
        col=1,
    )
    fig.update_xaxes(
        showticklabels=False, ticks="", showgrid=False, mirror=True, row=row, col=2
    )
    fig.update_yaxes(
        title_text=f"{channel_name.title()} Attribution",
        mirror=True,
        tickfont=dict(size=20),
        title_font=dict(size=20),
        range=[y_min - y_padding, y_max + y_padding],
        row=row,
        col=2,
    )


def plot_attributions_combined(
    smiles_list, model_name, n_steps=50, target_idx=None, channel_name=None
):
    model, tokenizer, device = load_model(model_name)
    target_idx = get_target_channel_idx(model, target_idx, channel_name)

    all_scores, all_tokens, all_kekule_smiles, predictions = compute_all_attributions(
        smiles_list, model, tokenizer, device, n_steps, target_idx
    )

    n_rows = len(smiles_list)
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        column_widths=[0.45, 0.55],
        specs=[[{"type": "image"}, {"type": "bar"}]] * n_rows,
        vertical_spacing=0.2 / max(n_rows, 1),
        horizontal_spacing=0.02,
    )

    for i, (smiles, scores, tokens, kekule_smiles) in enumerate(
        zip(smiles_list, all_scores, all_tokens, all_kekule_smiles)
    ):
        add_molecule_subplot(fig, i + 1, kekule_smiles, tokens, scores, channel_name)

    fig.update_layout(
        height=500 * n_rows,
        showlegend=False,
        width=2300,
        margin=dict(t=30, b=30, l=50, r=50),
        template="simple_white",
    )
    return fig, predictions


def save_attribution_plot(
    smiles_list,
    model_name,
    output_path,
    n_steps=150,
    target_idx=None,
    channel_name=None,
):
    fig, predictions = plot_attributions_combined(
        smiles_list, model_name, n_steps, target_idx, channel_name
    )
    fig.write_image(
        output_path, format="pdf", width=2300, height=500 * len(smiles_list)
    )

    basename = os.path.splitext(output_path)[0]
    with open(f"{basename}-predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {basename}-predictions.json")

    model, tokenizer, _ = load_model(model_name)
    idx = get_target_channel_idx(model, target_idx, channel_name)
    if idx is None and channel_name:
        raise ValueError(f"Channel '{channel_name}' not found in model channels")

    for i, smiles in enumerate(smiles_list):
        kekule_smiles, tokens, scores, _ = compute_attributions(
            smiles, model, tokenizer, n_steps, idx
        )
        norm, cmap = get_color_mapper(scores)
        structure_img = draw_molecule_with_attributions(
            kekule_smiles, tokens, scores, norm, cmap
        )
        if structure_img:
            mol_output_path = (
                f"{basename}-mol-{i}.pdf"
                if len(smiles_list) > 1
                else f"{basename}-mol.pdf"
            )
            structure_img.save(mol_output_path, "PDF", resolution=300.0)


if __name__ == "__main__":
    save_attribution_plot(
        ["CC(=O)OCC1=CC=CC=C1"],
        "mist-models/mist-26.9M-48kpooqf-odour",
        "fruity-attribution.pdf",
        channel_name="fruity",
    )
    save_attribution_plot(
        ["CC(=O)OCC1=CC=CC=C1"],
        "mist-models/mist-26.9M-48kpooqf-odour",
        "floral-attribution.pdf",
        channel_name="floral",
    )
