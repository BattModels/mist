---
language: en
library_name: transformers
license: gpl-3.0
tags:
  - mist
  - chemistry
  - molecular-property-prediction
---

# MIST: Molecular Insight SMILES Transformers

MIST is a family of molecular foundation models for molecular property prediction.
The models were pre-trained on SMILES strings from the [Enamine REAL Space](https://enamine.net/compound-collections/real-compounds/real-space-navigator) dataset using the Masked Language Modeling (MLM) objective, then fine-tuned for downstream prediction tasks.

## Model Details

- **Architecture**: Encoder-only transformer [``RoBERTa-PreLayerNorm``](https://huggingface.co/docs/transformers/en/model_doc/roberta-prelayernorm)
- **Pre-training**: Masked Language Modeling on molecular SMILES
- **Tokenization**: [``Smirk``](https://eeg.engin.umich.edu/smirk/) tokenizer


### Quick Start

```python
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained(
    "path/to/model",
    trust_remote_code=True
)

# Make predictions
smiles_batch = [
    "CCO",           # Ethanol
    "CC(=O)O",       # Acetic acid
    "c1ccccc1"       # Benzene
]
results = model.predict(smiles_batch)
```

### Setting Up Your Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note**: SMIRK tokenizers require Rust to be installed. See the [Rust installation guide](https://www.rust-lang.org/tools/install) for details.


## Model Inputs and Outputs

### Inputs
- **SMILES strings**: Standard SMILES notation for molecular structures
- **Batch size**: Variable, automatically padded during inference

### Outputs
- **Predictions**: Task-specific numerical or categorical predictions
- **Format**: Dictionary with channel names and predicted values (if channels are configured), or raw tensor output


## Provided Models

### Pre-trained
- `mist-1.8B-dh61satt`: Flagship MIST model (MIST-1.8B)
- `mist-28M-ti624ev1`: Smaller MIST model (MIST-28M).

Below is a full list of finetuned variants hosted on HuggingFace:
### MoleculeNet Benchmark Models

| Folder                       | Encoder  | Dataset                              |
| ---------------------------- | :------: | ------------------------------------ |
| mist-1.8B-fbdn8e35-bbbp      | MIST-1.8B| MoleculeNet BBBP                     |
| mist-1.8B-1a4puhg2-hiv       | MIST-1.8B| MoleculeNet HIV                      |
| mist-1.8B-m50jgolp-bace      | MIST-1.8B| MoleculeNet BACE                     |
| mist-1.8B-uop1z0dc-tox21     | MIST-1.8B| MoleculeNet Tox21                    |
| mist-1.8B-lu1l5ieh-clintox   | MIST-1.8B| MoleculeNet ClinTox                  |
| mist-1.8B-l1wfo7oa-sider     | MIST-1.8B| MoleculeNet SIDER.                   |
| mist-1.8B-hxiygjsm-esol      | MIST-1.8B| MoleculeNet ESOL                     |
| mist-1.8B-iwqj2cld-freesolv  | MIST-1.8B| MoleculeNet FreeSolv                 |
| mist-1.8B-jvt4azpz-lipo      | MIST-1.8B| MoleculeNet Lipophilicity            |
| mist-1.8B-8nd1ot5j-qm8       | MIST-1.8B| MoleculeNet QM8                      |
| mist-28M-3xpfhv48-bbbp       | MIST-28M | MoleculeNet BBBP                     |
| mist-28M-8fh43gke-hiv        | MIST-28M | MoleculeNet HIV                      |
| mist-28M-8loj3bab-bace       | MIST-28M | MoleculeNet BACE                     |
| mist-28M-kw4ks27p-tox21      | MIST-28M | MoleculeNet Tox21                    |
| mist-28M-97vfcykk-clintox    | MIST-28M | MoleculeNet ClinTox                  |
| mist-28M-z8qo16uy-sider      | MIST-28M | MoleculeNet SIDER                    |
| mist-28M-kcwb9le5-esol       | MIST-28M | MoleculeNet ESOL                     |
| mist-28M-0uiq7o7m-freesolv   | MIST-28M | MoleculeNet FreeSolv                 |
| mist-28M-xzr5ulva-lipo       | MIST-28M | MoleculeNet Lipophilicity            |
| mist-28M-gzwqzpcr-qm8        | MIST-28M | MoleculeNet QM8                      |
| mist-28M-kkgx0omx-qm9        | MIST-28M | MoleculeNet QM9                      |

#### QM9 Benchmark Models
The single target (MIST-1.8B encoder) models for properties in QM9 are available.

| Folder                       | Encoder  | Target                                                            |
| ---------------------------- | :------: | ----------------------------------------------------------------- |
| mist-1.8B-ez05expv-mu        | MIST-1.8B| μ - Dipole moment (unit: D)                                       |
| mist-1.8B-rcwary93-alpha     | MIST-1.8B| α - Isotropic polarizability (unit: Bohr^3)                       |
| mist-1.8B-jmjosq12-homo      | MIST-1.8B| HOMO - Highest occupied molecular orbital energy (unit: Hartree)  |
| mist-1.8B-n14wshc9-lumo      | MIST-1.8B| LUMO - Lowest unoccupied molecular orbital energy (unit: Hartree) |
| mist-1.8B-kayun6v3-gap       | MIST-1.8B| Gap - Gap between HOMO and LUMO (unit: Hartree)                   |
| mist-1.8B-xxe7t35e-r2        | MIST-1.8B| \<R2\> - Electronic spatial extent (unit: Bohr^2)                 |
| mist-1.8B-6nmcwyrp-zpve      | MIST-1.8B| ZPVE - Zero point vibrational energy (unit: Hartree)              |
| mist-1.8B-a7akimjj-u0        | MIST-1.8B| U0 - Internal energy at 0K (unit: Hartree)                        |
| mist-1.8B-85f24xkj-u298      | MIST-1.8B| U298 - Internal energy at 298.15K (unit: Hartree)                 |
| mist-1.8B-3fbbz4is-h298      | MIST-1.8B| H298 - Enthalpy at 298.15K (unit: Hartree)                        |
| mist-1.8B-09sntn03-g298      | MIST-1.8B| G298 - Free energy at 298.15K (unit: Hartree)                     |
| mist-1.8B-j356b3nf-cv        | MIST-1.8B| Cv - Heat capacity at 298.15K (unit: cal/(mol*K))                 |



### Finetuned Single Task Models

These models consist of a MIST-encoder and task network finetuned on a single dataset used in the applications demonstrated in the manuscript.

| Folder                    | Encoder  | Dataset                                                     |
| ------------------------- | :------: | ----------------------------------------------------------- |
| mist-26.9M-48kpooqf-odour | MIST-28M | Olfaction                                                   |
| mist-26.9M-6hk5coof-dn    | MIST-28M | Donor Number                                                |
| mist-26.9M-0vxdbm36-kt    | MIST-28M | Kamlet-Taft Solvochromatic Parameters                       |
| mist-26.9M-b302p09x-bp    | MIST-28M | Boiling Point (Part of Characteristic Temperatures Dataset) |
| mist-26.9M-cyuo2xb6-fp    | MIST-28M | Flash Point (Part of Characteristic Temperatures Dataset)   |
| mist-26.9M-y3ge5pf9-mp    | MIST-28M | Melting Point (Part of Characteristic Temperatures Dataset) |

### Finetuned Multi-Task Models
These are additional multi-target finetuned models consisting of a MIST encoder and task network.
| Folder                     | Encoder  | Dataset                                                     |
| -------------------------- | :------: | ----------------------------------------------------------- |
| mist-26.9M-kkgx0omx-qm9    | MIST-28M | QM9 Dataset with SMILES randomization                       |
| mist-28M-ttqcvt6fs-toxcast | MIST-28M | ToxCast                                                     |
| mist-28M-yr1urd2c-muv      | MIST-28M | Maximum Unbiased Validation (MUV)                           |

### Finetuned Mixture Models

These models consist of a MIST-encoder and physics informed task network for mixture property prediction.
| Folder                           | Encoder  | Dataset                                                     |
| -------------------------------- | :------: | ----------------------------------------------------------- |
| mist-conductivity-28M-2mpg8dcd   | MIST-28M | Ionic Conductivity                                          |
| mist-mixtures-zffffbex           | MIST-28M | Excess Density, Molar Volume and Molar Enthalpy             |

## Citation

If you use this model in your research, please cite:

```bibtex
@online{MIST,
  title = {Foundation Models for Discovery and Exploration in Chemical Space},
  author = {Wadell, Alexius and Bhutani, Anoushka and Azumah, Victor and Ellis-Mohr, Austin R. and Kelly, Celia and Zhao, Hancheng and Nayak, Anuj K. and Hegazy, Kareem and Brace, Alexander and Lin, Hongyi and Emani, Murali and Vishwanath, Venkatram and Gering, Kevin and Alkan, Melisa and Gibbs, Tom and Wells, Jack and Varshney, Lav R. and Ramsundar, Bharath and Duraisamy, Karthik and Mahoney, Michael W. and Ramanathan, Arvind and Viswanathan, Venkatasubramanian},
  date = {2025-10-20},
  eprint = {2510.18900},
  eprinttype = {arXiv},
  eprintclass = {physics},
  doi = {10.48550/arXiv.2510.18900},
  url = {http://arxiv.org/abs/2510.18900},  
}
```

## License and Notice

Model weights are provided as-is for research purposes only, without guarantees of correctness, fitness for purpose, or warranties of any kind.

**Restrictions:**
- Research use only
- No redistribution without permission
- No commercial use without licensing agreement

For questions, issues, or licensing inquiries, please contact [venkvis@umich.edu](mailto:venkvis@umich.edu).

<hr>
