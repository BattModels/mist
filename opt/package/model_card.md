# MIST: Molecular Insight SMILES Transformer This model was pre-trained on
SMILES string from [Enamine REAL Space](https://enamine.net/compound-collections/real-compounds/real-space-navigator)
dataset using the Masked Language Modeling (MLM) objective.


## Installation & Usage
Create a virtual environment and install the requirements:
```shell
python -m venv .venv
.venv/bin/pip install -r requirements.txt

```

> Smirk tokenizers may require rust to be installed to run.
> See [install rust](https://www.rust-lang.org/tools/install) for more info.

See [demo.py](demo.py) for an example of how to use the model, or run
`.venv/bin/python demo.py` to run the demo.

> The demo assumes pandas is installed (for pretty printing the results). You
> may need to install it with `.venv/bin/pip install pandas` before running the
> demo

# Notice
This is a pre-release version of MIST and has not undergone extensive
safety, validation, or performance testing. Model weights are provided as-is
without guarantees of correctness or fitness for purpose.

Use of this model is limited to research purposes only, and may not be
distributed or used for any other purpose.
