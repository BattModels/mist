"""Predict the original four discontinuity examples with a local odour model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prepare import local_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = local_model(args.model)
    smiles = ["CC(=O)C", "CC(=S)C", "CC1=CCC(CC1)C(=C)C", "CC1(C)OC2CCC1(C)CC2"]
    # Preserve the original ten evaluation-mode passes and mean logits.
    logits = sum(
        pd.DataFrame(
            {key: 0.1 * value["value"] for key, value in model.predict(smiles).items()},
            index=smiles,
        )
        for _ in range(10)
    )
    probabilities = 1 / (1 + np.exp(-logits))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(
        args.output_dir / "small_change_normalized_logits.csv", index_label="Column1"
    )
    (probabilities > 0.5).astype("int64").to_csv(
        args.output_dir / "small_change.csv", index_label="Column1"
    )


if __name__ == "__main__":
    main()
