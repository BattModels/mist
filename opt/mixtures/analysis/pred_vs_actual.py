from electrolyte_fm.data_modules import ComponentDataModule
import matplotlib.pyplot as plt
from lipari_cm import lipari10_cmap
from electrolyte_fm.models.model_utils import DeepSpeedMixin
import numpy as np

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
# plt.rcParams['lines.markersize'] = 2
# plt.rcParams['lines.linewidth'] = 1
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 5
plt.rcParams["axes.labelsize"] = 5
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["legend.fontsize"] = 4
plt.rcParams["font.family"] = "Serif"
plt.rcParams["grid.linewidth"] = 0.1
plt.rcParams["figure.dpi"] = 500
plt.rcParams["savefig.dpi"] = 500
plt.rcParams["mathtext.fontset"] = "stix"

run_id = "ur7v2a6y"  # "c7ssptg7" #
pretrained_ckpt = f"/home/abhutani/electrolyte-fm/mist/{run_id}/checkpoints/last.ckpt"
model = DeepSpeedMixin.load(pretrained_ckpt)

dm = ComponentDataModule(
    path="/home/abhutani/electrolyte-fm/diffmix_data/Ion_Cond_Tle20/",
    target_col="ln k",
    n_components=5,
    val_batch_size=20,
    tokenizer=pretrained_ckpt,
    include_temperature=True,
    iterable=False,
)

fig, ax = plt.subplots(figsize=(2, 2))

dm.setup(stage="test")
sc = None  # Initialize scatter plot in case of using outside loop later

for idx, batch in enumerate(dm.test_dataloader()):
    # Assuming temperature is already the actual value, no need to divide again if normalized is not required
    colors = batch[
        "temperature"
    ].numpy()  # Use unnormalized temperature directly for coloring
    sc = ax.scatter(
        np.exp(batch["target"].numpy()),
        np.exp(model(batch)[0].detach().numpy()),
        c=colors,
        cmap=lipari10_cmap,
    )
    if idx > 10:
        break

# Add the diagonal line
ax.plot([0, 14], [0, 14], c="black")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Set the spines and axes customization
ax.set_xticks(np.arange(0, 16, 2))
ax.set_yticks(np.arange(0, 16, 2))
ax.set_xticklabels(np.arange(0, 16, 2))
ax.set_yticklabels(np.arange(0, 16, 2))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-0.5, 14.5)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

# Create a colorbar with unnormalized temperature labels
cbar = plt.colorbar(sc, ax=ax)
# Set the label to reflect physical units like Kelvin or Celsius
cbar.set_label("Temperature [K]", rotation=-90, labelpad=15)
cbar.ax.yaxis.set_label_position("left")  # Align label position

# Rotate label to face inward
cbar.ax.yaxis.set_label_position("right")
cbar.ax.yaxis.label.set_rotation(270)

# Set the axis labels
ax.set_xlabel("AEM $\sigma$ [mS/cm]")
ax.set_ylabel("MIST $\sigma$ [mS/cm]")

plt.savefig(f"ionic_cond_{run_id}", bbox_inches="tight", dpi=500)
