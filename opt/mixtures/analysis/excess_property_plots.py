import numpy as np
import pandas as pd
import os
from matplotlib.lines import Line2D
import torch
import tqdm
from electrolyte_fm.data_modules import ComponentDataModule
import matplotlib.pyplot as plt
from electrolyte_fm.models.model_utils import DeepSpeedMixin
from lipari_cm import lipari10_cmap
from matplotlib.colors import Normalize
from itertools import product
import glob
from re import sub


def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 5
plt.rcParams["axes.labelsize"] = 5
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["font.family"] = "Serif"
plt.rcParams["grid.linewidth"] = 0.1
plt.rcParams["figure.dpi"] = 500
plt.rcParams["savefig.dpi"] = 500
plt.rcParams["mathtext.fontset"] = "stix"

excess_data_paths = glob.glob(
    "/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_**.csv"
)
solvents = pd.concat([pd.read_csv(fp) for fp in excess_data_paths])
solvents = solvents.sub2_name.unique()
# random.shuffle(solvents)
solvents = [s for s in solvents if not s.endswith("ate") and not s.endswith("ol")]
# np.append(solvents.sub1_name.unique(), solvents.sub2_name.unique())
symbols = list(Line2D.filled_markers)
symbols = list(product(Line2D.fillStyles, Line2D.filled_markers))
symbols = dict(zip(solvents, Line2D.filled_markers))


def interpolate_color(value, vmin, vmax):
    """
    Map a scalar value to a color using a colormap.

    Parameters:
    - value: float, the scalar value to map to a color.
    - vmin: float, the minimum value of the data range.
    - vmax: float, the maximum value of the data range.
    - cmap: str or Colormap, the colormap to use. Default is 'viridis'.

    Returns:
    - color: RGBA tuple, the color corresponding to the input value.
    """
    # Normalize the value within the given range
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Map the value to a color
    color = lipari10_cmap(norm(value))

    return color


def generate_data(
    source_dir: str, save_dir: str, target_col: str = "excess_molar_volume/(cm3/mol)"
):
    test_df = pd.read_csv(os.path.join(source_dir, "test.csv"))
    df_list = []
    for group, idx in test_df.groupby(
        [
            "sub1_name",
            "sub2_name",
            "smi1",
            "smi2",
            "temperature",
        ]
    ).groups.items():
        sub1_name, sub2_name, smi1, smi2, temperature = group
        x1 = np.linspace(0, 1, 50)
        x2 = 1 - x1
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
            }
        )
        df["sub1_name"] = sub1_name
        df["sub2_name"] = sub2_name
        df["smi1"] = smi1
        df["smi2"] = smi2
        df[target_col] = 0
        df["temperature"] = temperature
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv(os.path.join(save_dir, "test.csv"))
    test_df = test_df[df.columns]
    test_df.to_csv(os.path.join(save_dir, "train.csv"))
    test_df.to_csv(os.path.join(save_dir, "val.csv"))


def run_inference(
    pretrained_ckpt: str,
    data_dir: str,
    val_batch_size: int,
    target_col: str = "excess_molar_volume/(cm3/mol)",
):
    model = DeepSpeedMixin.load(pretrained_ckpt)
    model.to(torch.device("cpu"))
    model.eval()
    dm = ComponentDataModule(
        path=data_dir,
        target_col=target_col,
        n_components=2,
        val_batch_size=1,
        tokenizer=pretrained_ckpt,
        include_temperature=True,
    )
    dm.setup(stage="test")

    results = []
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    for idx, batch in tqdm.tqdm(enumerate(dm.test_dataloader())):
        pred = model.forward(batch, transform=True)

        results.extend(pred.flatten().tolist())
    test_df["predicted"] = results
    test_df.to_csv(os.path.join(data_dir, "inference.csv"))


def plot(
    properties: dict,
    plot_sub: str = "Propylene Carbonate",
):
    fig, axes = plt.subplots(2, 1, figsize=(2, 2), sharex=True)
    row = 0
    s2 = set()
    for k, v in properties.items():
        ax = axes[row]
        ax.grid()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray")
        data_dir = v["data_dir"]
        target_col = v["target_col"]
        axis_label = v["axis_label"]
        results_file = os.path.join(data_dir, "inference.csv")
        data_file = os.path.join(data_dir, "train.csv")

        df = pd.read_csv(results_file)
        df = df[df.sub1_name == plot_sub]
        df.reset_index(inplace=True)
        data_df = pd.read_csv(data_file)
        tmin = df.temperature.min()
        tmax = df.temperature.max()
        t2 = set()
        line_num = 0
        for group, idx in df.groupby(
            ["sub1_name", "sub2_name", "temperature"]
        ).groups.items():
            sub1_name, sub2_name, temperature = group
            # if line_num < 5 and temperature in t2:
            #     continue
            t2.add(temperature)

            _data_df = data_df[
                (data_df.sub2_name == sub2_name)
                # & (data_df.temperature == temperature)
            ]
            if sub2_name not in symbols:
                continue

            t2.add(temperature)
            _data_df = data_df[
                (data_df.sub2_name == sub2_name) & (data_df.temperature == temperature)
            ]

            _df = df.iloc[idx]
            _df = _df.sort_values("x1")
            color = interpolate_color(temperature, tmin, tmax)
            ax.plot(_df.x1, _df.predicted, color=color, lw=1, zorder=1)
            ax.scatter(
                _data_df.x1,
                _data_df[target_col],
                marker=symbols[sub2_name],
                label=f"{sub2_name.title()}" if sub2_name not in s2 else None,
                edgecolors="black",
                facecolors=color,
                linewidths=0.5,
                zorder=100,
            )
            s2.add(sub2_name)
            line_num += 1

        # Axis titles and legend
        if row > 0:
            ax.set_xlabel(
                "$x_1$",
                labelpad=0.01,
            )
        ax.set_ylabel(
            axis_label,
            labelpad=0.01,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        # ax.set_ylim(-1.0, 1.)
        ax.set_xlim(-0.1, 1.1)
        ax.tick_params(axis="both", width=0.1)
        row += 1

    fig.legend(
        handlelength=0,
        ncol=3,
        frameon=False,
        fontsize=4,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.19),
    )
    # Create a single colorbar for the entire figure
    norm = Normalize(tmin, tmax)
    sm = plt.cm.ScalarMappable(cmap=lipari10_cmap, norm=norm)
    colorbar_ax = fig.add_axes([0.05, -0.05, 0.95, 0.03])  # x, y, width, height
    cbar = plt.colorbar(sm, cax=colorbar_ax, orientation="horizontal")
    cbar.set_label("Temperature [K]", fontsize=5)
    cbar.outline.set_linewidth(0.2)
    cbar.solids.set_rasterized(True)
    cbar.ax.tick_params(width=0.2)
    plt.savefig(f"{camel_case(plot_sub)}.png", bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    properties = {
        "enthalpy": {
            "data_dir": "plot_excess_molar_enthalpy",
            "target_col": "excess_molar_enthalpy/(J/mol)",
            "run_id": "apfyl2ld",
            "axis_label": "$H_E$ [J/mol]",
        },
        "volume": {
            "data_dir": "plot_excess_molar_volume",
            "target_col": "excess_molar_volume/(cm3/mol)",
            "run_id": "vt5f3jcm",
            "axis_label": "$V_m$ [cm3/mol]",
        },
    }

    for k, v in properties.items():
        # generate_data(
        #     source_dir=f"/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_{k}",
        #     save_dir=v["data_dir"],
        #     target_col=v["target_col"],
        # )
        # pretrained_ckpt = (
        #     f"/home/abhutani/electrolyte-fm/mist/{v['run_id']}/checkpoints/last.ckpt"
        # )
        # run_inference(
        #     pretrained_ckpt=pretrained_ckpt,
        #     data_dir=v['data_dir'],
        #     val_batch_size=2,
        #     target_col=v['target_col']
        # )
        plot(properties)
