import numpy as np
import pandas as pd
import itertools
import os

from typing import List, Union
import torch
import tqdm
from electrolyte_fm.data_modules import ComponentDataModule
import matplotlib.pyplot as plt
import niceplots
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from lipari_cm import lipari10_cmap
from matplotlib.colors import Normalize

plt.style.use(niceplots.get_style())
plt.rcParams["figure.constrained_layout.use"] = False


def softmax_rowwise(arr):
    """
    Computes the softmax function for each row of a NumPy array.

    Args:
      arr: A 2D NumPy array.

    Returns:
      A NumPy array with the same shape as arr, where each row is the softmax of the
      corresponding row in arr.
    """
    # Subtract the maximum value in each row for numerical stability
    row_maxes = arr.max(axis=1, keepdims=True)
    shifted_arr = arr - row_maxes

    # Calculate the exponential of each element
    exp_arr = np.exp(shifted_arr)

    # Divide by the sum of exponentials in each row
    row_sums = exp_arr.sum(axis=1, keepdims=True)
    return exp_arr / row_sums


def get_triangular_grid(n=31, prec=1e-6):
    """Triangular grid

    Parameters
    ----------
    n : int, optional
        Number of grid points along one ternary axis, by default 11
    prec : float, optional
        Tolerance for triangular points, by default 1e-6

    Returns
    -------
    (t, l, r) : tuple[np.ndarray]
        Ternary coordinates.
    """
    # top axis in descending order to start from the top point
    t = np.linspace(1, 0, n)
    points = []
    for tmp in itertools.product(t, repeat=3):
        if abs(sum(tmp) - 1.0) > prec:
            continue
        points.append(tmp)
    points = np.array(points)
    return points


def generate_dataset(
    salt_mole_fraction: float = 0.1,
    save_dir: str = "ternary",
    solvents: List[str] = ["C1COC(=O)O1", "COC(=O)OC", "CC1COC(=O)O1"],
    salt: str = "LiPF6",
):
    os.makedirs(save_dir, exist_ok=True)
    compositions = (1 - salt_mole_fraction) * get_triangular_grid()
    df = pd.DataFrame(
        {
            "x1": compositions[:, 0],
            "x2": compositions[:, 1],
            "x3": compositions[:, 2],
            "x4": salt_mole_fraction * np.ones_like(compositions[:, 0]),
            "x5": salt_mole_fraction * np.ones_like(compositions[:, 0]),
        }
    )
    for i in range(1, 4):
        df[f"smi{i}"] = solvents[i - 1]
    if salt == "LiPF6":
        df["smi4"] = "F[P-](F)(F)(F)(F)F"
    elif salt == "TFSI":
        df["smi4"] = "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"
    df["smi5"] = "[Li+]"
    df["temperature"] = 273.15
    df["target"] = 0
    df.to_csv(os.path.join(save_dir, "train.csv"))
    df.to_csv(os.path.join(save_dir, "val.csv"))
    df.to_csv(os.path.join(save_dir, "test.csv"))


def transferance(pred):
    return pred[-2]


def conductivity(pred):
    return pred[-3]


def product(pred):
    return pred[-2] * pred[-3]


def diffusioncoeff(pred):
    return pred[-4]


targets = {
    "transferance": transferance,
    "conductivity": conductivity,
    "product": product,
    "diffusioncoeff": diffusioncoeff,
}

target_names = {
    "transferance": "$t_+$",
    "conductivity": "$\sigma [mS]$",
    "product": "$t_+ \cdot \sigma$",
    "diffusioncoeff": "$\mathcal{D}_{Li} [cm^2/s]$",
}


def run_inference(
    pretrained_ckpt: str, data_dir: str, val_batch_size: int, target: str
):
    target_fn = targets[target]
    model = DeepSpeedMixin.load(pretrained_ckpt)
    model.to(torch.device("cpu"))
    model.eval()
    dm = ComponentDataModule(
        path=data_dir,
        target_col="target",
        n_components=5,
        val_batch_size=val_batch_size,
        tokenizer=pretrained_ckpt,
        include_temperature=True,
    )
    dm.setup(stage="train")

    results = []
    dx = []

    for idx, batch in tqdm.tqdm(enumerate(dm.test_dataloader())):
        for i in range(3):
            batch[f"composition_{i}"].requires_grad = True

        pred = model(batch)[0]
        _dx = torch.autograd.grad(
            [target_fn(pred[i, :]) for i in range(pred.shape[0])],
            [batch[f"composition_{i}"] for i in range(3)],
            retain_graph=True,
        )

        _dx = torch.stack(_dx, dim=1)
        for bdx in range(pred.shape[0]):
            dx.append(_dx[bdx, :].tolist())
            results.append(target_fn(pred[bdx, :]).item())

    v = np.array(results)
    c = get_triangular_grid()
    x1 = c[:, 0]
    x2 = c[:, 1]
    x3 = c[:, 2]
    dx = np.array(dx)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "transferance": v,
            "dx1": dx[:, 0],
            "dx2": dx[:, 1],
            "dx3": dx[:, 2],
        }
    )
    inference_filename = os.path.basename(data_dir)
    inference_filename, extension = os.path.splitext(inference_filename)
    print(f"Saving inference data to {inference_filename}")
    df.to_csv(f"{target}_{inference_filename}.csv")


def cartesian_to_barycentric(d_cartesian):
    """
    Converts a gradient vector from Cartesian coordinates to barycentric coordinates.

    Parameters:
    - d_cartesian: A 2D numpy array where each row represents a vector (dx, dy, dz)
      in Cartesian coordinates.

    Returns:
    - np.ndarray: Corresponding rows of vectors in barycentric coordinates.
    """

    if len(d_cartesian.shape) != 2 or d_cartesian.shape[1] != 3:
        raise ValueError("Input should be a 2D array with shape (n, 3).")

    # Assuming d_cartesian comes in the form of n x 3 array (dx, dy, dz)
    d_barycentric = np.zeros_like(d_cartesian)

    # Convert each vector
    for i, (dx, dy, dz) in enumerate(d_cartesian):
        # Calculate the change in barycentric coordinates
        d_barycentric[i, 0] = dx - (dy + dz) / 2
        d_barycentric[i, 1] = dy - (dx + dz) / 2
        d_barycentric[i, 2] = dz - (dx + dy) / 2

    return d_barycentric


def plot_vector_fields(data_files: Union[List, str], target: str):
    if isinstance(data_files, str):
        data_files = [
            data_files,
        ]

    n_files = len(data_files)
    n_cols = max(1, int(0.5 * n_files))
    fig, axes = plt.subplots(
        2, n_cols, figsize=(3 * n_files, 9.6), subplot_kw={"projection": "ternary"}
    )
    fig.subplots_adjust(left=0.075, right=0.85, wspace=0.7, hspace=0.5)

    data_files = sorted(data_files, key=lambda x: x.split("_")[1])

    # Collect transferance values to determine global min and max
    all_transferance = []

    for data_file in data_files:
        df = pd.read_csv(data_file)
        all_transferance.extend(df.transferance)

    normalize = Normalize(vmin=min(all_transferance), vmax=max(all_transferance))

    for i, data_file in enumerate(data_files):
        _, salt, composition = data_file.split("_")
        salt = "LiPF_6" if salt == "LiPF6" else "LiTFSI"
        composition = float(composition.split(".")[0])
        col = int(composition // 5) - 1
        composition = 1e-2 * composition
        composition = "$x_{" + salt + "}$" + f"= {composition:.2f}"
        row = i // n_cols

        ax = axes[row, col] if n_files > 1 else axes

        df = pd.read_csv(data_file)

        cs = ax.tricontour(
            df.x1,
            df.x2,
            df.x3,
            df.transferance,
            levels=50,
            colors="#989C97",
            linewidths=0.5,
            zorder=1,
        )

        max_bound = 0.98
        min_bound = 0.02

        df = df[(df.x1 < max_bound) & (df.x1 > min_bound)]
        df = df[(df.x2 < max_bound) & (df.x2 > min_bound)]
        df = df[(df.x3 < max_bound) & (df.x3 > min_bound)]

        dx = cartesian_to_barycentric(df[["dx1", "dx2", "dx3"]].values)

        cs = ax.quiver(
            df.x1,
            df.x2,
            df.x3,
            dx[:, 0],
            dx[:, 1],
            dx[:, 2],
            df.transferance,
            units="dots",
            width=8,
            zorder=2,
            # scale=1,
            # scale_units="xy",
            cmap=lipari10_cmap,
            norm=normalize,
        )

        ax.set_tlabel("PC", fontweight="bold")
        ax.set_llabel("DMC", fontweight="bold")
        ax.set_rlabel("EC", fontweight="bold")
        ax.set_title(composition, pad=10, fontweight="bold")

    # Add a common color bar outside of the loop, aligned to the right of all plots
    cbar_axis = fig.add_axes([0.94, 0.1, 0.015, 0.8])  # Adjust position as needed
    cbar = fig.colorbar(cs, cax=cbar_axis)
    cbar.ax.set_title(target_names[target], pad=10, fontweight="bold")

    plt.savefig(f"{target}_salt_composition.png")
