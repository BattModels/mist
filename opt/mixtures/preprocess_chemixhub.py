import os
from pathlib import Path

import pandas as pd


def process_drug_solubility(
    raw_datapath: str = "drug-solubility/raw_data/DrugSolubilityData.csv",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath)
    df.fillna("NONE", inplace=True)
    df["total"] = df.comp_0 + df.comp_1 + df.comp_2
    df["temperature"] = df["Temperature, K"]
    for i in range(3):
        df[f"x{i+1}"] = df[f"comp_{i}"] / df.total
        df[f"smi{i+1}"] = df[f"SMILES_{i}"]
    grouped = df.groupby("Train_Test_Label")

    for category_name, group_df in grouped:
        filename = os.path.join(output_dir, f"{category_name}.csv")
        group_df.to_csv(filename, index=False)


def process_nist_logV(
    raw_datapath: str = "nist-logV/processed_data/processed_NISTlogV.csv",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath.parent / "compounds.csv")
    compound_lookup = dict(zip(df.compound_id, df.smiles))
    df = pd.read_csv(raw_datapath)
    df.fillna("NONE", inplace=True)

    df["x1"] = df.cmp_mole_fractions.apply(lambda x: eval(x)[0])
    df["x2"] = df.cmp_mole_fractions.apply(lambda x: eval(x)[-1])
    df["cmp1"] = df.cmp_ids.apply(lambda x: eval(x)[0])
    df["cmp2"] = df.cmp_ids.apply(lambda x: eval(x)[-1])
    df["smi1"] = df.cmp1.map(compound_lookup)
    df["smi2"] = df.cmp2.map(compound_lookup)
    df["temperature"] = df["T"]

    df_train = df.sample(frac=0.8)
    df_train.to_csv(os.path.join(output_dir, "train.csv"))
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.5, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"))
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"))


if __name__ == "__main__":
    process_drug_solubility()
    process_nist_logV()
