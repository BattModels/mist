from electrolyte_fm.models.model_utils import DeepSpeedMixin
from electrolyte_fm.utils.tokenizer import load_tokenizer
from transformers import PreTrainedTokenizerFast
from electrolyte_fm.models import LMFinetuning
from torch.nn import Module
from selfies import encoder as sf_encoder
from smirk import SmirkTokenizerFast
from torch import tensor
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops, MolFromSmiles


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


def _pool_pick_attention(attentions, layer, head, pool):
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
    model, tokenizer, tokenizer_name = maybe_load_model_and_tokenizer(
        checkpoint, tokenizer
    )

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
    tick_fontsize: int = 12,
    tokenizer: Optional[str] = None,
):
    """
    Plots attention maps for specified layers and heads of a model checkpoint with given SMILES string.

    Parameters:
    - checkpoint (str): Path to the model checkpoint.
    - smiles (str): SMILES string of the molecule.
    - pool (str, optional): Pooling method to apply, default is 'mean'.
    - layers (Optional[Union[int, List[int]]], optional): Layers to plot. Default is None, which plots the first layer.
    - heads (Optional[Union[int, List[int]]], optional): Attention heads to plot. Default is None, which plots the first head.
    - symmetrical (bool, optional): If True, makes the attention map symmetrical. Default is False.
    - tick_fontsize (int, optional): Font size of the plot ticks. Default is 12.
    - tokenizer (Optional[str], optional): Tokenizer to use. Default is None.

    Returns:
    - plt.Figure: The matplotlib figure with the plotted attention maps.
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

    # Add one additional row and column for labels
    fig, axarr = plt.subplots(
        nrows=nrows, ncols=ncols, squeeze=False, figsize=(6 * ncols, 6 * nrows)
    )

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
            filtered_attention = filtered_attention / filtered_attention.max()
            img1 = axarr[row_idx][col_idx].imshow(filtered_attention, aspect="equal")
            axarr[row_idx][col_idx].set_xticks(range(len(filtered_tokens)))
            axarr[row_idx][col_idx].set_yticks(range(len(filtered_tokens)))
            axarr[row_idx][col_idx].set_yticklabels(
                filtered_tokens, fontsize=tick_fontsize - 1
            )
            axarr[row_idx][col_idx].set_xticklabels(
                filtered_tokens, rotation="vertical", fontsize=tick_fontsize - 1
            )
            if row_idx == 0:
                axarr[row_idx][col_idx].set_title(
                    f"Head {head}", fontsize=tick_fontsize + 5
                )
            if col_idx == col_idx // 2:
                axarr[row_idx][col_idx].set_title(
                    f"Layer {layer}", fontsize=tick_fontsize + 5
                )

    # Additional row for 3D distances and bond matrix
    row_idx += 1
    col_idx += 1

    # Calculate 3D distances and plot inverse
    mol = MolFromSmiles(smiles)
    confid = AllChem.EmbedMolecule(mol)
    dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)
    dist_matrix = 1 / dist_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = dist_matrix / dist_matrix.max()  # normalize distance matrix
    bonds_mat, elem_tokens = get_bonds_mat(smiles)

    # Inverse 3D Distance
    axarr[row_idx][1].imshow(dist_matrix, aspect="equal")
    axarr[row_idx][1].set_xticks(range(len(elem_tokens)))
    axarr[row_idx][1].set_xticklabels(elem_tokens, fontsize=tick_fontsize - 1)
    axarr[row_idx][1].set_yticks(range(len(elem_tokens)))
    axarr[row_idx][1].set_yticklabels(elem_tokens, fontsize=tick_fontsize - 1)
    axarr[row_idx][1].set_title("Inverse 3D Distance", fontsize=tick_fontsize + 5)

    # Bond matrix
    axarr[row_idx][2].imshow(bonds_mat, aspect="equal")
    axarr[row_idx][2].set_xticks(range(len(elem_tokens)))
    axarr[row_idx][2].set_yticks(range(len(elem_tokens)))
    axarr[row_idx][2].set_xticklabels(
        elem_tokens, rotation="vertical", fontsize=tick_fontsize
    )
    axarr[row_idx][2].set_yticklabels(elem_tokens, fontsize=tick_fontsize)
    axarr[row_idx][2].set_title("Bond Matrix", fontsize=tick_fontsize + 5)

    # Format and adjust plot
    fig.tight_layout(pad=3.0)
    cbar_ax = fig.add_axes([0.05, -0.08, 0.6, 0.05])
    cbar = fig.colorbar(img1, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=tick_fontsize - 2)
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
    # try:
    highlight_struc = [mol.GetSubstructMatch(MolFromSmiles(s)) for s in subs]
    # except:
    #     highlight_struc = None
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
        with open(os.path.join(save_dir, "structure.png"), "wb") as png:
            png.write(drawing)


def mol_from_attention(
    checkpoint: str,
    smiles: str,
    save_dir: str,
    pool: str = "max",
    layer: Optional[Union[int, List[int]]] = None,
    head: Optional[int] = None,
    tokenizer: Optional[str] = None,
):
    """
    Reconstructs a molecular graph from an attention map and saves the resulting graph as an image.

    Parameters:
    - checkpoint (str): Path to the model checkpoint.
    - smiles (str): SMILES string of the molecule.
    - save_dir (str): Directory to save the resulting image.
    - pool (str, optional): Pooling method to apply to the attention maps. Default is 'max'.
    - layer (Optional[Union[int, List[int]]], optional): Layer from which to get the attention map. Default is None.
    - head (Optional[int], optional): Attention head to use. Default is None.
    - tokenizer (Optional[str], optional): Tokenizer to use. Default is None.
    """

    attentions, tokens = _get_attention_map_and_tokens(checkpoint, smiles, tokenizer)
    attention = _pool_pick_attention(attentions, layer, head, pool)
    attention = attention / attention.max()

    # Make the attention matrix symmetrical
    attention = (attention + attention.T) * 0.5
    np.fill_diagonal(attention, 0.0)

    # Binarize the attention matrix
    threshold = 0.5
    connectivity_matrix = (attention > threshold).astype(int)

    ignore_tokens = _non_element_tokens() if "_non_element_tokens" in globals() else []
    filtered_attention, filtered_tokens = filter_tokens(
        attention, tokens, ignore_tokens
    )
    filtered_connectivity, _ = filter_tokens(connectivity_matrix, tokens, ignore_tokens)

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

    for i in range(len(filtered_connectivity)):
        for j in range(i + 1, len(filtered_connectivity)):
            if filtered_connectivity[i][j] > 0:
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)

    mol = mol.GetMol()

    # Generate image
    highlight_struc = [atom_indices]
    drawing = Draw.MolToImage(mol, size=(500, 500), highlightAtoms=highlight_struc[0])
    output_file = os.path.join(save_dir, f"connectivity_{head}_{layer}.png")
    drawing.save(output_file)
    print(f"Saved molecule image to: {output_file}")

    return mol


if __name__ == "__main__":
    import os

    run_ids = ["q2egf8f2", "28znv46w", "3q7skx88", "nimbixvu", "pdess2id"]
    # smiles = "C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1"
    # smiles = "NN"
    smiles = "C1C=CC(COC)C=C1"
    # mol = MolFromSmiles(smiles)
    # for x in mol.GetAtoms():
    #     print(x.GetIdx(), x.GetHybridization())
    # rdDetermineBonds.DetermineBondOrders(mol, charge=0, embedChiral=False)
    # for bond in mol.GetBonds():
    #     print(bond.GetBondType())
    # smiles = "C(=CCl)Cl"
    # smiles = "C1CC2C=CC1C(C2Cl)Cl"

    for run_id in run_ids[:1]:
        tokenizer = None
        if "pdess2id" == run_id:
            tokenizer = "smirk-selfies"
        elif ("3q7skx88" == run_id) or ("nimbixvu" == run_id):
            tokenizer = "/home/abhutani/electrolyte_fm/smirk-gpe/smirk-gpe-50k-nmb-ss"
        chkpt = f"/nfs/turbo/coe-venkvis/mist/{run_id}/checkpoints/last.ckpt"
        tok = load_tokenizer("smirk-selfies")

        # for layer in range(0, 18, 3):
        #     ld_path = f"./layer_{layer}"
        #     os.makedirs(ld_path, exist_ok=True)

        #     # fig = plot_attention_map(checkpoint=chkpt, smiles=smiles, symmetrical=False, layers=layer, pool="mean", tokenizer=tokenizer)
        #     fig.savefig(os.path.join(ld_path, f"./test_{run_id}.pdf"), format='pdf', bbox_inches="tight",)
        #     plot_max_attention(checkpoint=chkpt, save_dir = ld_path, smiles=smiles, head = 1, layer=layer,tokenizer=tokenizer)
        # mol_from_attention(
        #         checkpoint=chkpt,
        #         smiles=smiles,
        #         save_dir=ld_path,
        #         pool = "max",
        #         layer = layer,
        #         head= 1,
        #         tokenizer= tokenizer,
        # )
        fig = plot_attention_map(
            checkpoint=chkpt,
            symmetrical=False,
            smiles=smiles,
            layers=list(range(0, 18, 3)),
            heads=list(range(0, 12)),
            pool="mean",
            tokenizer=tokenizer,
        )
        fig.savefig(
            f"./test_{run_id}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
