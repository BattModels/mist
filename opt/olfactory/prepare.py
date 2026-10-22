"""Produce olfaction prediction tables for the existing paper plots.

Model loading is local-only. Benchmark tasks supply fresh seed-42 splits and
validation-selected single-molecule and mixture models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing olfaction input at {path}")
    frame = pd.read_csv(path)
    return frame.rename(columns={"nonStereoSMILES": "smiles"})


def channels_from(path: Path) -> list[str]:
    channels = [
        channel["name"] for channel in yaml.safe_load(path.read_text())["channels"]
    ]
    if len(channels) != 135 or len(set(channels)) != 135:
        raise ValueError("The paper odour head requires 135 distinct channels")
    return channels


def local_model(path: Path):
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Supply a staged MIST odour model directory at {path}")
    import torch

    from electrolyte_fm.models.prod_finetune import load_finetuned

    model = load_finetuned(str(path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def descriptor_predictions(args) -> None:
    from electrolyte_fm.data_modules.utils import MolEncoding

    channels = channels_from(args.contract)
    frame = read_frame(args.input)
    if frame.empty or "smiles" not in frame or frame.smiles.isna().any():
        raise ValueError("Prediction input requires nonempty smiles rows")
    smiles = [MolEncoding("smiles-kekule")(value) for value in frame.smiles]
    if any(value is None for value in smiles):
        raise ValueError(
            "Prediction input contains invalid SMILES; no rows were dropped"
        )
    model = local_model(args.model)
    output = model.predict(smiles, batch_size=args.batch_size)
    if set(output) != set(channels):
        raise ValueError("Model channels disagree with the 135-channel odour contract")
    values = {name: np.asarray(output[name]["value"], dtype=float) for name in channels}
    predictions = pd.DataFrame({"smiles": frame.smiles, **values})
    if not np.isfinite(predictions[channels].to_numpy()).all():
        raise ValueError("Nonfinite odour predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)


def performance_tables(args) -> None:
    from functional_groups import identify_functional_groups
    from sklearn.metrics import roc_auc_score

    channels = channels_from(args.contract)
    train, test, predictions = map(
        read_frame, (args.train, args.test, args.predictions)
    )
    if train.empty or test.empty:
        raise ValueError("Train and test splits must both be nonempty")
    if test.smiles.tolist() != predictions.smiles.tolist():
        raise ValueError("Predictions must preserve the exact test molecule order")
    truth = test[channels].to_numpy(dtype=float)
    logits = predictions[channels].to_numpy(dtype=float)
    if not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("Held-out odour targets must be finite binary labels")
    if not np.isfinite(logits).all():
        raise ValueError("Held-out predictions must be finite logits")
    by_descriptor = [
        {
            "descriptor": name,
            "num_samples": int(train[name].sum()),
            "auroc": roc_auc_score(truth[:, index], logits[:, index]),
        }
        for index, name in enumerate(channels)
        if len(np.unique(truth[:, index])) == 2
    ]
    by_molecule = [
        {
            "smiles": smile,
            "functional_group": ";".join(identify_functional_groups(smile)),
            "auroc": roc_auc_score(target, prediction),
        }
        for smile, target, prediction in zip(test.smiles, truth, logits, strict=True)
        if len(np.unique(target)) == 2
    ]
    if not by_descriptor or not by_molecule:
        raise ValueError(
            "No evaluable odour channels or molecules in the test split"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_descriptor).to_csv(
        args.output_dir / "dataset_size_auroc.csv", index=False
    )
    pd.DataFrame(by_molecule).to_csv(
        args.output_dir / "functional_group_errors.csv", index=False
    )


def redjade_mapping(args) -> None:
    public = read_frame(args.public_mapping)
    supplied = read_frame(args.mapping)
    triplets = read_frame(args.triplets)
    required = set(triplets["RedJade Code A"]) | set(triplets["RedJade Code B"])
    mapping = dict(zip(public["RedJade Code"], public.SMILES, strict=True))
    columns = ["redjade_code", "canonical_isomeric_smiles", "source_or_release_id"]
    if not set(columns) <= set(supplied):
        raise ValueError(f"Supplied RedJade mapping requires {columns}")
    if supplied[columns].isna().any().any() or supplied.redjade_code.duplicated().any():
        raise ValueError(
            "Supplied RedJade mappings require unique codes and complete provenance"
        )
    for row in supplied.itertuples():
        if not str(row.source_or_release_id).strip():
            raise ValueError("RedJade identities require a source_or_release_id")
        if (
            row.redjade_code in mapping
            and mapping[row.redjade_code] != row.canonical_isomeric_smiles
        ):
            raise ValueError(f"Conflicting identity for {row.redjade_code}")
        mapping[row.redjade_code] = row.canonical_isomeric_smiles
    missing = required - mapping.keys()
    if missing:
        raise ValueError(
            f"Missing original RedJade identities: {', '.join(sorted(missing))}"
        )
    from rdkit import Chem

    if any(Chem.MolFromSmiles(mapping[code]) is None for code in required):
        raise ValueError("Invalid RedJade molecular identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"Sample Identifier": code, "SMILES": mapping[code]}
            for code in sorted(required)
        ]
    ).to_csv(args.output, index=False)


def mixture_predictions(args) -> None:
    # This reuses the existing MixtureSimilarityModel and MultiMixtureDataModule;
    # all checkpoint/tokenizer paths are explicit artifacts, never Hub fallbacks.
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing fitted mixture odour checkpoint at {args.checkpoint}"
        )
    if not (args.encoder / "config.json").is_file():
        raise FileNotFoundError(f"Supply the staged MIST encoder at {args.encoder}")
    frames = [
        read_frame(args.split_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    ]
    for frame in frames:
        for column in ("mix1_smiles", "mix2_smiles"):
            for value in frame[column]:
                molecules = json.loads(value)
                if (
                    not isinstance(molecules, list)
                    or not molecules
                    or not all(isinstance(x, str) and x for x in molecules)
                ):
                    raise ValueError(
                        f"{column} requires a nonempty JSON list of SMILES"
                    )
        if (
            frame.empty
            or not np.isfinite(frame.value).all()
            or not frame.value.between(0, 1).all()
        ):
            raise ValueError(
                "Mixture targets must be nonempty finite distances in [0, 1]"
            )
    import torch

    from electrolyte_fm.data_modules import MultiMixtureDataModule
    from electrolyte_fm.models import MixtureSimilarityModel

    data = MultiMixtureDataModule(
        path=str(args.split_dir),
        tokenizer=str(args.encoder),
        target_columns=["value"],
        encoding="smiles-kekule",
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=0,
    )
    data.prepare_data()
    data.setup("test")
    model = MixtureSimilarityModel.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        encoder_ckpt=str(args.encoder),
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    target, predicted = [], []
    with torch.inference_mode():
        for batch in data.test_dataloader():
            batch = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
            prediction, _ = model(batch, transform=False)
            predicted.extend(prediction[:, 0].float().cpu().tolist())
            target.extend(batch["target"][:, 0].float().cpu().tolist())
    if len(target) != len(frames[-1]) or not np.isfinite(predicted).all():
        raise ValueError(
            "Mixture prediction lost held-out rows or produced nonfinite values"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": target, "prediction": predicted}).to_csv(
        args.output, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    for name in ("model", "input", "contract", "output"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=32)
    predict.set_defaults(run=descriptor_predictions)
    performance = commands.add_parser("performance")
    for name in ("train", "test", "predictions", "contract", "output-dir"):
        performance.add_argument(f"--{name}", type=Path, required=True)
    performance.set_defaults(run=performance_tables)
    mapping = commands.add_parser("redjade")
    for name in ("public-mapping", "mapping", "triplets", "output"):
        mapping.add_argument(f"--{name}", type=Path, required=True)
    mapping.set_defaults(run=redjade_mapping)
    mixture = commands.add_parser("mixture")
    for name in ("checkpoint", "encoder", "split-dir", "output"):
        mixture.add_argument(f"--{name}", type=Path, required=True)
    mixture.add_argument("--batch-size", type=int, default=4)
    mixture.set_defaults(run=mixture_predictions)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
