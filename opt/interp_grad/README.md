# Token Attribution Visualization

Visualizes which parts of a molecule contribute most to model predictions using Layer Integrated Gradients.

## What it Does

Creates attribution plots showing:
- **Molecular structure** with atoms/bonds colored by attribution scores
- **Token bar charts** showing individual SMILES token contributions
- **Predictions** saved as JSON for each molecule

Red indicates negative attribution (decreases prediction), green indicates positive attribution (increases prediction).

## Installation

```bash
uv sync
```

## Usage

### Basic Example

```python
from token_attribution import save_attribution_plot

save_attribution_plot(
    smiles_list=["CC(=O)OCC1=CC=CC=C1"],
    model_name="mist-models/mist-26.9M-48kpooqf-odour",
    output_path="attribution.pdf",
    channel_name="fruity",
    n_steps=150
)
```

### Parameters

- `smiles_list`: List of SMILES strings to analyze
- `model_name`: HuggingFace model name or local path
- `output_path`: Output PDF path (generates `{basename}.pdf`, `{basename}-mol.pdf`, `{basename}-predictions.json`)
- `channel_name`: Target property channel (e.g., "fruity", "floral")
- `n_steps`: Integration steps (default: 150, higher = more accurate but slower)


## How it Works

1. Converts SMILES to Kekulé form
2. Tokenizes using SMIRK tokenizer
3. Computes attributions using [Layer Integrated Gradients](https://captum.ai/api/layer.html#layer-integrated-gradients) (using a `[PAD]` token as the baseline).
4. Maps token attributions to molecular atoms/bonds
5. Generates visualizations with RDKit and Plotly
