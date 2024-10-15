from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer
from electrolyte_fm.models import LMFinetuning
from selfies import encoder as sf_encoder
from torch import tensor
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolTransforms


def _non_element_tokens():
    CHIRAL = ["@", "@@"]
    CHIRAL_CONFIG = ["TH", "AL", "SP", "TB", "OH"]
    BONDS = [".", "-", "=", "#", "$", ":", "/", "\\"]
    DIGITS = [str(x) for x in range(10)]
    SELFIES_STRUC = ["Ring", "Branch"]
    return CHIRAL + CHIRAL_CONFIG + BONDS + DIGITS + SELFIES_STRUC


def get_bonds_mat(smiles):
    """
    Get the bond matrix for a SMILES string using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    bonds = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol.GetBonds()]
    bonds_mat = np.zeros((len(mol.GetAtoms()), len(mol.GetAtoms())))

    for ele in bonds:
        bonds_mat[ele[0], ele[1]] = 1
        bonds_mat[ele[1], ele[0]] = 1

    bond_tokens = []

    for atom in mol.GetAtoms():
        bond_tokens.append(atom.GetSymbol())

    return bonds_mat, bond_tokens


def filter_tokens(attention_map, tokens, ignore_tokens):
    """
    Remove unwanted tokens from a SMILES string and
    corrensponding rows and columns from attention map.
    """
    wanted_idx, wanted_tokens = [], []
    ignore_tokens = set(ignore_tokens)
    for idx, token in enumerate(tokens):
        if token in ignore_tokens:
            continue
        else:
            wanted_idx.append(idx)
            wanted_tokens.append(token)

    return attention_map[wanted_idx, :][:, wanted_idx], wanted_tokens


def plot_attention_map(
    checkpoint: str,
    smiles: str,
    layer: Optional[Union[int, List]] = None,
    head: Optional[int] = None,
    symmetrical: bool = True,
    tick_fontsize: int = 40,
    tokenizer: Optional[str] = None,
):
    # Load model and tokenizer from checkpoint
    model = DeepSpeedMixin.load(checkpoint)
    if isinstance(model, LMFinetuning):
        encoder = model.encoder
    else:
        encoder = model.model
    if tokenizer:
        tok = load_tokenizer(tokenizer)
    else:
        tok = load_tokenizer(checkpoint)

    # Get tokens and attention map
    seq = sf_encoder(smiles) if (tokenizer and "selfies" in tokenizer) else smiles
    encoding = tok(
        [
            seq,
        ]
    )
    tokens = tok.tokenize(seq)
    if "special_tokens_mask" in encoding:
        encoding.pop("special_tokens_mask")
    encoding = {k: tensor(v) for k, v in encoding.items()}

    # `attentions` is a tuple with length = number of hidden layers
    # each element has shape [1, num_attention_heads, seq_len, seq_len]
    attentions = encoder(**encoding, output_attentions=True).attentions

    if not (layer or head):
        print("Layer and head not specified, plotting mean attention.")
        # Mean pooled attention map with shape [seq_len, seq_len]
        attention = tensor(
            [a[0].mean(axis=0).detach().numpy() for a in attentions]
        ).mean(axis=0)
    elif layer:
        if isinstance(layer, list):
            attention = tensor(
                [
                    a[0].mean(axis=0).detach().numpy()
                    for a in attentions[layer[0] : layer[-1]]
                ]
            ).mean(axis=0)
        else:
            attention = attentions[layer][0].mean(axis=0).detach()
    elif head:
        attention = tensor([a[0][head].detach().numpy() for a in attentions]).mean(
            axis=0
        )
    else:
        attention = attentions[layer][0][head].detach()

    if symmetrical:
        attention = (attention + attention.transpose(0, 1)) * 0.5
        attention.fill_diagonal_(0.0)

    ignore_tokens = _non_element_tokens()
    filtered_attention, filtered_tokens = filter_tokens(
        attention, tokens, ignore_tokens
    )

    # Plot
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    img1 = axarr[0].imshow(filtered_attention, aspect="equal")
    axarr[0].set_xticks(range(0, len(filtered_tokens)))
    axarr[0].set_yticks(range(0, len(filtered_tokens)))
    axarr[0].set_yticklabels(filtered_tokens, fontsize=tick_fontsize - 1)
    axarr[0].set_xticklabels(
        filtered_tokens, rotation="vertical", fontsize=tick_fontsize - 1
    )
    axarr[0].set_title("Avg-Pooled Attention", fontsize=tick_fontsize + 5)

    # COL 2: calculate 3D distances and plot inverse
    mol = Chem.MolFromSmiles(smiles)
    confid = AllChem.EmbedMolecule(mol)
    dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)
    dist_matrix = 1 / dist_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = dist_matrix / dist_matrix.max()  # normalize distance matrix
    axarr[1].imshow(dist_matrix, aspect="equal")
    axarr[1].set_xticks(range(0, len(filtered_tokens)))
    axarr[1].set_xticklabels(filtered_tokens, fontsize=tick_fontsize - 1)
    axarr[1].set_yticks(range(0, len(filtered_tokens)))
    axarr[1].set_yticklabels(filtered_tokens, fontsize=tick_fontsize - 1)
    axarr[1].set_title("Inverse 3D Distance", fontsize=tick_fontsize + 5)

    # COL 3: get bond matrix and plot
    bonds_mat, gold_tokens = get_bonds_mat(smiles)
    axarr[2].imshow(bonds_mat, aspect="equal")
    axarr[2].set_xticks(range(0, len(gold_tokens)))
    axarr[2].set_yticks(range(0, len(gold_tokens)))
    axarr[2].set_xticklabels(gold_tokens, rotation="vertical", fontsize=tick_fontsize)
    axarr[2].set_yticklabels(gold_tokens, fontsize=tick_fontsize)
    axarr[2].set_title("Bond Matrix", fontsize=tick_fontsize + 5)

    # Format and adjust plot
    fig.tight_layout(pad=3.0)
    cbar_ax = fig.add_axes([0.05, -0.08, 0.6, 0.05])
    cbar = fig.colorbar(img1, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=tick_fontsize - 2)
    fig.suptitle(f"{smiles}", fontsize=tick_fontsize + 5, y=1.15)
    return fig
