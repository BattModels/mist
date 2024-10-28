#!/usr/bin/env python
import logging
import argparse
import shutil
from pathlib import Path
from typing import Optional
from electrolyte_fm.utils.ckpt import SaveConfigWithCkpts
from electrolyte_fm.utils.tokenizer import load_tokenizer
from importlib.metadata import version

logging.basicConfig(level=logging.INFO)

README = """
# MIST: Molecular Insight SMILES Transformer
This model was pre-trained on SMILES string from [Enamine REAL Space](https://enamine.net/compound-collections/real-compounds/real-space-navigator)
dataset using the Masked Language Modeling (MLM) objective. It has not been fine-tuned on any specific task.

## Using the model
Mist can be loaded using HuggingFace's `from_pretrained` method:

```python
from transformers import AutoModel, AutoTokenizer
from smirk import SmirkTokenizerFast
AutoTokenizer.register(
    "SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast 
)

model = AutoModel.from_pretrained("/path/to/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("/path/to/{model_name}")

# Tokenzie some molecules
molecules = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccc(O)cc1)N)[C@H]3SC2(C)C",
    "[NH3][Pt@SP2](Cl)(Cl)[NH3]",
]
batch = tokenizer(molecules)

# Get predictions
model = model.to("cuda")
out = model(torch.tensor(batch["input_ids"]).to(model.device))

```

Further information on how to use HuggingFace models can be found at [huggingface.co/transformers](https://huggingface.co/transformers).

# Smirk
A pre-release version of [smirk](https://arxiv.org/abs/2409.15370) has been included with this model.
As a pre-release version bugs, incomplete features, and other issues may exist. While we 
expect that final released version of smirk will be similar, we cannot guarantee that it will be identical.
Smirk is licensed under the Apache License 2.0 (see the bundled LICENSE file for details).

# Notice
This is a pre-release version of MIST and has not undergone extensive safety, validation, or performance testing.
Model weights are provided as-is without guarantees of correctness or fitness for purpose.

Use of this model is limited to research purposes only, and may not be distributed or used for any other purpose.

"""

def write_requirements(save_directory: Path):
    with open(save_directory.joinpath("requirements.txt"), "w") as fid:
        for dep in ["transformers", "torch"]:
            fid.write(f"{dep}=={version(dep)}\n")

        smirk_version = version("smirk")
        smirk_sdist = Path(".cache", "smirk", f"smirk-{smirk_version}.tar.gz")
        if smirk_sdist.exists():
            shutil.copy(smirk_sdist, save_directory)
            fid.write(smirk_sdist.name + "\n")
        else:
            fid.write(f"smirk-{smirk_version}.tar.gz\n")
            logging.warning("user must manually bundle smirk")



def export_model(checkpoint_dir: str, config_path: Optional[str] = None):
    checkpoint_dir = Path(checkpoint_dir)
    model = SaveConfigWithCkpts.load(checkpoint_dir, config_path)
    model_name = checkpoint_dir.parent.parent.name
    save_directory = Path(f"mist-{model_name}")
    save_directory.mkdir(exist_ok=True, parents=True)
    write_requirements(save_directory)

    # Save the encoder
    model.model.save_pretrained(
        save_directory=save_directory,
        safe_serialization=True,
        push_to_hub=False,
    )

    # Save the tokenizer
    tokenizer = load_tokenizer(str(checkpoint_dir.resolve()))
    tokenizer.save_pretrained(save_directory, legacy_format=False, push_to_hub=False)

    # Include README
    with open(save_directory.joinpath("README.md"), "w") as fid:
        fid.write(README.format_map({"model_name": save_directory.name}))

    if Path("LICENSE").exists():
        shutil.copy("LICENSE", save_directory)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    export_model(args.checkpoint_dir, args.config_path)
