import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from excess_property_plots import interpolate_color
from lipari_cm import lipari10_cmap
from matplotlib.colors import Normalize

from electrolyte_fm.data_modules import ComponentDataModule
from electrolyte_fm.models.model_utils import DeepSpeedMixin

plt.style.use("./mist.mplstyle")


def plot(solvent_name, pretrained_ckpt, model, base_dir):
    """
    Plot ionic conductivity as a function of mole fraction.
    """

    fig, axes = plt.subplots(1, 2, figsize=(4, 1), sharey=True)

    salts = ["LiPF6", "LiTFSI"]

    for sidx, salt_name in enumerate(salts):
        ax = axes[sidx]
        dir_list = glob.glob(os.path.join(base_dir, f"{solvent_name}_{salt_name}_*"))

        for cidx, directory in enumerate(dir_list):
            temperature = re.search(r"_([0-9]+)$", directory).group(1)
            color = interpolate_color(float(temperature), 243, 293)
            solvent_name = directory.split("/")[-1].split("_")[0]
            dm = ComponentDataModule(
                path=directory,
                target_col="Cond(mS)2",
                n_components=5,
                val_batch_size=20,
                tokenizer=pretrained_ckpt,
                include_temperature=True,
            )

            dm.setup(stage="test")

            for idx, batch in enumerate(dm.test_dataloader()):
                pred = model(batch)[0].detach().numpy()
                pred = np.exp(pred)
                if pred.max() > 100:
                    continue
                else:
                    ax.plot(
                        batch["composition_4"].numpy(),
                        pred,
                        label=solvent_name,
                        color=color,
                    )

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_xlabel("$x_{Li}$")
        if sidx < 1:
            ax.set_ylabel("$\sigma$ [mS/cm]")
        ax.set_xticks(np.around(np.linspace(0, 0.2, 10), 1))
        ax.set_xticklabels(np.around(np.linspace(0, 0.2, 10), 1))

    # Create a single colorbar for the entire figure
    norm = Normalize(243, 293)
    sm = plt.cm.ScalarMappable(cmap=lipari10_cmap, norm=norm)
    colorbar_ax = fig.add_axes([0.05, -0.02, 0.9, 0.05])  # x, y, width, height
    cbar = plt.colorbar(sm, cax=colorbar_ax, orientation="horizontal")
    cbar.set_label("Temperature [K]")
    cbar.solids.set_rasterized(True)
    cbar.ax.tick_params(width=0.2)

    # Save figure
    plt.savefig(f"ic_solvent_{solvent_name}.png", bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    run_id = "ur7v2a6y"  # "z4ni8hcj" # "c7ssptg7"
    pretrained_ckpt = (
        f"/home/abhutani/electrolyte-fm/mist/{run_id}/checkpoints/last.ckpt"
    )
    model = DeepSpeedMixin.load(pretrained_ckpt)
    solvent_name = "PC"
    plot(solvent_name, pretrained_ckpt, model)
