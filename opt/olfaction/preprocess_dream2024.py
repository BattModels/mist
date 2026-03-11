"""Preprocess DREAM 2024 olfactory similarity data for MultiMixtureDataModule."""

import json
from pathlib import Path
import pandas as pd
import pubchempy as pcp


def get_smiles_from_cid(cid: int, cache: dict) -> str | None:
    """Get SMILES from PubChem CID with caching."""
    if cid in cache:
        return cache[cid]
    try:
        compounds = pcp.get_compounds(cid, "cid")
        if compounds:
            cache[cid] = compounds[0].isomeric_smiles
            return cache[cid]
    except Exception as e:
        print(f"Warning: Could not retrieve SMILES for CID {cid}: {e}")
    cache[cid] = None
    return None


def load_mixture_definitions(filepath: Path) -> dict:
    """Load mixture definitions: mixture_label -> list of CIDs."""
    df = pd.read_csv(filepath).fillna(0)
    mixture_dict = {}
    for _, row in df.iterrows():
        cids = [
            int(row[col])
            for col in df.columns
            if col.startswith("CID") and int(row[col]) > 0
        ]
        mixture_dict[row["Mixture Label"]] = cids
    return mixture_dict


def load_cid_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded CID cache: {len(cache)} entries")
        return cache
    return {}


def fetch_missing_smiles(all_cids: set, cid_cache: dict, cache_file: Path):
    missing_cids = [cid for cid in all_cids if cid not in cid_cache]
    if missing_cids:
        print(f"Fetching SMILES for {len(missing_cids)} CIDs from PubChem...")
        for cid in missing_cids:
            get_smiles_from_cid(cid, cid_cache)
        with open(cache_file, "w") as f:
            json.dump(cid_cache, f, indent=2)

    failed = [cid for cid in all_cids if not cid_cache.get(cid)]
    if failed:
        print(f"Warning: Failed to retrieve {len(failed)} CIDs: {sorted(failed)}")


def find_target_column(df: pd.DataFrame) -> str:
    for col in [
        "Experimental Values",
        "Predicted_Experimental_Values",
        "Experimental values",
    ]:
        if col in df.columns:
            return col
    raise ValueError("No target column found")


def process_mixture_pairs(
    df: pd.DataFrame, mixture_dict: dict, cid_cache: dict
) -> list:
    """Process DataFrame of mixture pairs into SMILES format."""
    mix1_col = "Mixture 1" if "Mixture 1" in df.columns else "Mixture_1"
    mix2_col = "Mixture 2" if "Mixture 2" in df.columns else "Mixture_2"
    target_col = find_target_column(df)

    processed = []
    for _, row in df.iterrows():
        mix1_cids = mixture_dict.get(row[mix1_col], [])
        mix2_cids = mixture_dict.get(row[mix2_col], [])
        mix1_smiles = [cid_cache[cid] for cid in mix1_cids if cid_cache.get(cid)]
        mix2_smiles = [cid_cache[cid] for cid in mix2_cids if cid_cache.get(cid)]

        if mix1_smiles and mix2_smiles:
            processed.append(
                {
                    "mix1_smiles": mix1_smiles,
                    "mix2_smiles": mix2_smiles,
                    "mix1_cids": mix1_cids,
                    "mix2_cids": mix2_cids,
                    "target": row[target_col],
                }
            )
    return processed


def save_train_val_test(data: pd.DataFrame, output_dir: Path, train_frac: float):
    data["mix1_smiles"] = data["mix1_smiles"].apply(str)
    data["mix2_smiles"] = data["mix2_smiles"].apply(str)

    train = data.sample(frac=train_frac, random_state=42)
    remaining = data.drop(train.index)
    train.to_csv(output_dir / "train.csv", index=False)

    if train_frac == 0.8:
        val = remaining.sample(frac=0.5, random_state=42)
        test = remaining.drop(val.index)
        val.to_csv(output_dir / "val.csv", index=False)
        test.to_csv(output_dir / "test.csv", index=False)
        print(
            f"Saved {len(train)} train, {len(val)} val, {len(test)} test to {output_dir}"
        )
    else:
        remaining.to_csv(output_dir / "val.csv", index=False)
        print(f"Saved {len(train)} train, {len(remaining)} val to {output_dir}")


def process_dream2024_data(
    data_dir: str = "/nfs/turbo/coe-venkvis/abhutani/mist-dream/data/dream_2024",
    output_dir: str = "dream2024_processed",
):
    """Process DREAM 2024 data for MultiMixtureDataModule."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "cid_to_smiles.json"
    cid_cache = load_cid_cache(cache_file)

    train_mixtures = load_mixture_definitions(
        data_dir / "Cleaned_Mixure_Definitions_Training_Set.csv"
    )
    leaderboard_mixtures = load_mixture_definitions(
        data_dir / "Mixure_Definitions_Leaderboard_set.csv"
    )
    print(
        f"Loaded {len(train_mixtures)} training, {len(leaderboard_mixtures)} leaderboard mixtures"
    )

    all_cids = set()
    for cids in list(train_mixtures.values()) + list(leaderboard_mixtures.values()):
        all_cids.update(cids)

    fetch_missing_smiles(all_cids, cid_cache, cache_file)

    train_df = pd.read_csv(data_dir / "TrainingData_mixturedist.csv")
    train_processed = process_mixture_pairs(train_df, train_mixtures, cid_cache)

    leaderboard_df = pd.read_csv(data_dir / "Leaderboard_set_Submission_form.csv")
    leaderboard_processed = process_mixture_pairs(
        leaderboard_df, leaderboard_mixtures, cid_cache
    )

    train_df = pd.DataFrame(train_processed)
    train_df["mix1_smiles"] = train_df["mix1_smiles"].apply(str)
    train_df["mix2_smiles"] = train_df["mix2_smiles"].apply(str)

    train_split = train_df.sample(frac=0.9, random_state=42)
    val_split = train_df.drop(train_split.index)
    train_split.to_csv(output_dir / "train.csv", index=False)
    val_split.to_csv(output_dir / "val.csv", index=False)

    test_df = pd.DataFrame(leaderboard_processed)
    test_df["mix1_smiles"] = test_df["mix1_smiles"].apply(str)
    test_df["mix2_smiles"] = test_df["mix2_smiles"].apply(str)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"Saved {len(train_split)} train, {len(val_split)} val, {len(test_df)} test to {output_dir}"
    )


def process_olfboost_data(
    data_dir: str = "/nfs/turbo/coe-venkvis/abhutani/mist-dream/data/OlfBoost",
    output_dir: str = "OlfBoost_processed",
):
    """
    Process DREAM 2024 OlfBoost augemented data for MultiMixtureDataModule.
    https://github.com/Satarifard/CWYK-Olfboost/tree/0c6ae49d1d1343226e96733263e479ddcad79e7d/data/processed
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "cid_to_smiles.json"
    cid_cache = load_cid_cache(cache_file)

    train_mixtures = load_mixture_definitions(
        data_dir / "Mixure_Definitions_augmented_dataset.csv"
    )
    print(f"Loaded {len(train_mixtures)} mixtures")

    all_cids = set(cid for cids in train_mixtures.values() for cid in cids)
    fetch_missing_smiles(all_cids, cid_cache, cache_file)

    train_df = pd.read_csv(data_dir / "gt_with_dataset_V2_augmented_dataset.csv")
    train_processed = process_mixture_pairs(train_df, train_mixtures, cid_cache)

    save_train_val_test(pd.DataFrame(train_processed), output_dir, train_frac=0.8)


def process_satarifard2025_test_set(
    data_dir: str = "/nfs/turbo/coe-venkvis/abhutani/mist-dream/data/Satarifard2025",
    output_dir: str = "Satarifard2025_processed",
):
    """
    Process DREAM 2024 test set for MultiMixtureDataModule as released in
    https://github.com/Satarifard/DREAM-olfactory-mixtures-prediction-challenge/tree/c26be61362270b425275beeeff686a11d2812ed3/Test_Dataset
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "cid_to_smiles.json"
    cid_cache = load_cid_cache(cache_file)

    test_mixtures = load_mixture_definitions(
        data_dir / "Test_set_Mixure_Definitions.csv"
    )
    print(f"Loaded {len(test_mixtures)} mixtures")

    all_cids = set(cid for cids in test_mixtures.values() for cid in cids)
    fetch_missing_smiles(all_cids, cid_cache, cache_file)

    test_df = pd.read_csv(data_dir / "Test_set_mixturedist.csv")
    test_processed = process_mixture_pairs(test_df, test_mixtures, cid_cache)

    test_df = pd.DataFrame(test_processed)
    test_df["mix1_smiles"] = test_df["mix1_smiles"].apply(str)
    test_df["mix2_smiles"] = test_df["mix2_smiles"].apply(str)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved {len(test_df)} test examples to {output_dir}")


if __name__ == "__main__":
    process_satarifard2025_test_set()
