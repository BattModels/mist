import os
from typing import List, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, MolFromSmiles, rdmolops
from selfies import encoder as sf_encoder
from smirk import SmirkTokenizerFast
from torch import tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerFast

from electrolyte_fm.models import LMFinetuning
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ModuleNotFoundError:
    print("Attention Map analysis requires plotting depency matplotlib.")


def _non_element_tokens():
    CHIRAL = ["@", "@@"]
    CHIRAL_CONFIG = ["TH", "AL", "SP", "TB", "OH"]
    BONDS = [".", "-", "=", "#", "$", ":", "/", "\\"]
    DIGITS = [str(x) for x in range(10)]
    BRACKETS = ["(", ")"]
    SELFIES_STRUC = ["Ring", "Branch"]
    return CHIRAL + CHIRAL_CONFIG + BONDS + DIGITS + BRACKETS + SELFIES_STRUC


def get_bonds_mat(smiles):
    """
    Get the bond matrix for a SMILES string using RDKit.
    """
    mol = MolFromSmiles(smiles)
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


def _pool_pick_attention(attentions, layer, head, pool=None):
    if not (layer or head):
        if pool == "mean" or "avg":
            attention = tensor(
                [a[0].mean(axis=0).detach().numpy() for a in attentions]
            ).mean(axis=0)
        elif pool == "max":
            attention = tensor(
                [a[0].max(axis=0).detach().numpy() for a in attentions]
            ).mean(axis=0)
    elif layer:
        if isinstance(layer, list):
            if pool == "mean" or "avg":
                attention = tensor(
                    [
                        a[0].mean(axis=0).detach().numpy()
                        for a in attentions[layer[0] : layer[-1]]
                    ]
                ).mean(axis=0)
            elif pool == "max":
                attention = tensor(
                    [a[0].max(axis=0).detach().numpy() for a in attentions]
                )
        else:
            if pool == "mean" or "avg":
                attention = attentions[layer][0].mean(axis=0).detach()
            elif pool == "max":
                attention = tensor(
                    [a[0].max(axis=0).detach().numpy() for a in attentions]
                )
    elif head:
        if pool == "mean" or "avg":
            attention = tensor([a[0][head].detach().numpy() for a in attentions]).mean(
                axis=0
            )
        elif pool == "max":
            attention = tensor([a[0][head].detach().numpy() for a in attentions]).max(
                axis=0
            )
    else:
        attention = attentions[layer][0][head].detach()
    return attention


def maybe_load_model_and_tokenizer(
    checkpoint: Union[str, Module],
    tokenizer: Optional[Union[str, PreTrainedTokenizerFast, SmirkTokenizerFast]] = None,
):
    """
    Load model and or tokenizer from checkpoint if not already loaded.
    """
    # Load model if checkpoint path passed
    if isinstance(checkpoint, str):
        model = DeepSpeedMixin.load(checkpoint)
    # Already loaded
    elif isinstance(checkpoint, Module):
        model = checkpoint

    # Already loaded
    if isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(
        tokenizer, SmirkTokenizerFast
    ):
        tok = tokenizer
        tokenizer_name = str(tok.__class__)
    # Load tokenizer
    elif tokenizer and isinstance(tokenizer, str):
        tokenizer_name = tokenizer
        tok = load_tokenizer(tokenizer)
    elif tokenizer is None and isinstance(checkpoint, str):
        tok = load_tokenizer(checkpoint)
        tokenizer_name = str(tok.__class__)

    return model, tok, tokenizer_name


def _get_attention_map_and_tokens(
    checkpoint: Union[str, Module],
    smiles: str,
    tokenizer: Optional[Union[str, PreTrainedTokenizerFast]] = None,
):
    model, tok, tokenizer_name = maybe_load_model_and_tokenizer(checkpoint, tokenizer)

    if isinstance(model, LMFinetuning):
        encoder = model.encoder
    else:
        encoder = model.model

    # Get tokens and attention map
    seq = (
        sf_encoder(smiles)
        if (tokenizer_name and "selfies" in tokenizer_name)
        else smiles
    )
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
    return attentions, tokens


def plot_attention_map(
    checkpoint: str,
    smiles: str,
    pool: str = "mean",
    layers: Optional[Union[int, List]] = None,
    heads: Optional[Union[int, List]] = None,
    symmetrical: bool = False,
    tick_fontsize: int = 18,
    tokenizer: Optional[str] = None,
):
    """
    Plots attention maps for specified layers and heads of a model checkpoint with given SMILES string.
    """

    checkpoint, tokenizer, tokenizer_name = maybe_load_model_and_tokenizer(
        checkpoint, tokenizer
    )

    if isinstance(heads, int) or heads is None:
        heads = [heads]

    if isinstance(layers, int) or layers is None:
        layers = [layers]

    nrows = len(layers) + 1 if layers else 2
    ncols = len(heads) if heads else 1

    fig = plt.figure(figsize=(6 * ncols, 6 * (nrows - 1) + 12))
    outer_gs = GridSpec(nrows, ncols, figure=fig)

    for row_idx, layer in enumerate(layers):
        for col_idx, head in enumerate(heads):
            attentions, tokens = _get_attention_map_and_tokens(
                checkpoint, smiles, tokenizer
            )
            attention = _pool_pick_attention(attentions, layer, head, pool)

            if symmetrical:
                attention = (attention + attention.transpose(0, 1)) * 0.5
                attention.fill_diagonal_(0.0)

            ignore_tokens = []  # _non_element_tokens()
            filtered_attention, filtered_tokens = filter_tokens(
                attention, tokens, ignore_tokens
            )

            # Plot attention map
            inner_gs = outer_gs[row_idx, col_idx]
            ax = fig.add_subplot(inner_gs)
            filtered_attention = filtered_attention / filtered_attention.max()
            img = ax.imshow(filtered_attention, aspect="equal")
            ax.set_xticks(range(len(filtered_tokens)))
            ax.set_yticks(range(len(filtered_tokens)))
            ax.set_yticklabels(filtered_tokens, fontsize=tick_fontsize - 1)
            ax.set_xticklabels(
                filtered_tokens, rotation="vertical", fontsize=tick_fontsize - 1
            )
            if row_idx == 0:
                ax.set_title(f"Head {head}", fontsize=tick_fontsize + 5)
                if col_idx == ncols // 2:
                    ax.set_title(
                        f"Layer {layer}\nHead {head}", fontsize=tick_fontsize + 5
                    )
            elif col_idx == ncols // 2:
                ax.set_title(f"Layer {layer}", fontsize=tick_fontsize + 5)

    # Additional row for 3D distances and bond matrix
    inner_gs2 = outer_gs[-1, :].subgridspec(1, 2)

    # Calculate 3D distances and plot inverse
    mol = MolFromSmiles(smiles)
    confid = AllChem.EmbedMolecule(mol)
    dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)
    dist_matrix = 1 / dist_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = dist_matrix / dist_matrix.max()  # normalize distance matrix
    bonds_mat, elem_tokens = get_bonds_mat(smiles)

    # Inverse 3D Distance
    ax_dist = fig.add_subplot(inner_gs2[0])
    ax_dist.imshow(dist_matrix, aspect="equal")
    ax_dist.set_xticks(range(len(elem_tokens)))
    ax_dist.set_xticklabels(elem_tokens, fontsize=tick_fontsize)
    ax_dist.set_yticks(range(len(elem_tokens)))
    ax_dist.set_yticklabels(elem_tokens, fontsize=tick_fontsize)
    ax_dist.set_title("Inverse 3D Distance", fontsize=tick_fontsize + 5)

    # Bond matrix
    ax_bond = fig.add_subplot(inner_gs2[1])
    ax_bond.imshow(bonds_mat, aspect="equal")
    ax_bond.set_xticks(range(len(elem_tokens)))
    ax_bond.set_yticks(range(len(elem_tokens)))
    ax_bond.set_xticklabels(elem_tokens, rotation="vertical", fontsize=tick_fontsize)
    ax_bond.set_yticklabels(elem_tokens, fontsize=tick_fontsize)
    ax_bond.set_title("Bond Matrix", fontsize=tick_fontsize + 5)

    # Format and adjust plot
    fig.tight_layout(h_pad=0.0, w_pad=1.5)
    left, bottom, width, height = 1.0, 0.25, 0.03, 0.65
    cbar_ax = fig.add_axes([left, bottom, width, height])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=tick_fontsize + 10)
    fig.suptitle(f"{smiles}", fontsize=tick_fontsize + 5, y=1.00)

    return fig


def plot_max_attention(
    checkpoint: str,
    smiles: str,
    save_dir: str,
    pool: str = "max",
    layer: Optional[int] = None,
    head: Optional[int] = None,
    tick_fontsize: int = 40,
    tokenizer: Optional[str] = None,
):
    mol = MolFromSmiles(smiles)
    assert mol, "Unable to get molecule from SMILES"

    attentions, tokens = _get_attention_map_and_tokens(checkpoint, smiles, tokenizer)
    attention = _pool_pick_attention(attentions, layer, head, pool)
    ignore_tokens = _non_element_tokens()

    filtered_attention, filtered_tokens = filter_tokens(
        attention, tokens, ignore_tokens
    )

    # locate maximum attention
    i, j = np.unravel_index(
        np.argmax(filtered_attention, axis=None), filtered_attention.shape
    )

    # highlight substructure
    subs = [filtered_tokens[i], filtered_tokens[j]]

    highlight_struc = [mol.GetSubstructMatch(MolFromSmiles(s)) for s in subs]
    if highlight_struc and i != j:
        drawing = Draw.MolsToGridImage(
            [
                mol,
            ],
            subImgSize=(500, 500),
            highlightAtomLists=highlight_struc,
            returnPNG=True,
        )
        with open(
            os.path.join(save_dir, f"max_attention_layer_{layer}.png"), "wb"
        ) as png:
            png.write(drawing)
    else:
        drawing = Draw.MolsToGridImage(
            [
                mol,
            ],
            subImgSize=(500, 500),
            returnPNG=True,
        )
    return drawing


def mol_from_attention(
    checkpoint: str,
    smiles: str,
    save_dir: str,
    layer: Optional[Union[int, List[int]]] = None,
    head: Optional[int] = None,
    tokenizer: Optional[str] = None,
):
    """
    Reconstructs a molecular graph from an attention map and saves the resulting graph as an image.
    """

    attentions, tokens = _get_attention_map_and_tokens(checkpoint, smiles, tokenizer)

    attention = _pool_pick_attention(attentions, layer, head)
    attention = attention / attention.max()

    # Make the attention matrix symmetrical
    attention = (attention + attention.transpose(0, 1)) * 0.5
    attention.fill_diagonal_(0.0)

    # TODO: better mechanism mapping of attention to bond type
    # Binarize the attention matrix
    ignore_tokens = _non_element_tokens()
    filtered_attention, filtered_tokens = filter_tokens(
        attention, tokens, ignore_tokens
    )
    threshold = 0.5
    connectivity_matrix = (abs(filtered_attention) > threshold).numpy().astype(int)

    # Construct molecule from connectivity matrix
    mol = Chem.RWMol()
    atom_indices = []
    atom_map = {}
    for i, token in enumerate(filtered_tokens):
        if token not in ignore_tokens:
            atom = Chem.Atom(token)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)
            atom_map[i] = idx

    for i in range(len(connectivity_matrix)):
        for j in range(i + 1, len(connectivity_matrix)):
            if connectivity_matrix[i][j] > 0:
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)

    mol = mol.GetMol()

    # Generate image
    highlight_struc = [atom_indices]
    drawing = Draw.MolToImage(mol, size=(500, 500), highlightAtoms=highlight_struc[0])

    return drawing
