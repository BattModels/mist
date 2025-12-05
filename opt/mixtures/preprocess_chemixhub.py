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


def process_binary_mixtures(
    raw_datapath: str = "nist-logV/processed_data/processed_NISTlogV.csv",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath.parent / "compounds.csv")
    compound_lookup = dict(zip(df.compound_id, df.smiles))
    df = pd.read_csv(raw_datapath)

    df["x1"] = df.cmp_mole_fractions.apply(lambda x: eval(x)[0])
    df.x1.fillna(0, inplace=True)
    df["x2"] = df.cmp_mole_fractions.apply(lambda x: eval(x)[-1])
    df.x2.fillna(0, inplace=True)
    df["cmp1"] = df.cmp_ids.apply(lambda x: eval(x)[0])
    df["cmp2"] = df.cmp_ids.apply(lambda x: eval(x)[-1])
    df["smi1"] = df.cmp1.map(compound_lookup)
    df.smi1.fillna("NONE", inplace=True)
    df["smi2"] = df.cmp2.map(compound_lookup)
    df.smi2.fillna("NONE", inplace=True)
    if "T" in df.columns:
        df["temperature"] = df["T"]

    df_train = df.sample(frac=0.7, random_state=42)
    df_train.to_csv(os.path.join(output_dir, "train.csv"))
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.33, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"))
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"))


def process_miscible_solvents(
    raw_datapath: str = "miscible-solvent/raw_data/MiscibleSolventData.csv",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath)
    max_comp = 5

    for comp_num in range(max_comp):
        df[f"comp_{comp_num}"].fillna(0, inplace=True)
    df["total"] = df.comp_0 + df.comp_1 + df.comp_2 + df.comp_3 + df.comp_4
    for comp_num in range(max_comp):
        df[f"x{comp_num + 1}"] = df[f"comp_{comp_num}"] / df.total
        df[f"smi{comp_num + 1}"] = df[f"SMILES_{comp_num}"]
        df[f"smi{comp_num + 1}"].fillna("NONE", inplace=True)
    df_train = df.sample(frac=0.7, random_state=42)
    df_train.to_csv(os.path.join(output_dir, "train.csv"))
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.33, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"))
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"))


def process_MON(raw_datapath: str = "MON/raw_data/published_MONdata.csv"):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath)

    max_comp = 121
    for comp_num in range(max_comp):
        df[f"x{comp_num + 1}"] = df[f"Mole fraction of cmp{comp_num}"] * 1e-2
        df[f"x{comp_num + 1}"].fillna(0, inplace=True)
        df[f"smi{comp_num + 1}"] = df[f"cmp_{comp_num}_smiles"]
        df[f"smi{comp_num + 1}"].fillna("NONE", inplace=True)
    df_train = df.sample(frac=0.7, random_state=42)
    df_train.to_csv(os.path.join(output_dir, "train.csv"))
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.33, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"))
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"))


def maybe_get_index(x, comp_num):
    try:
        x = eval(x)[comp_num]
    except IndexError:
        x = None
    return x


def process_il_thermo(
    raw_datapath: str = "ionic-liquids/processed_data/processed_IlThermoData.csv",
    prop: str = "Viscosity",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name, prop)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath.parent / "compounds.csv")
    compound_lookup = dict(zip(df.compound_id, df.smiles))
    df = pd.read_csv(raw_datapath)
    df = df[df.property == prop]

    max_comp = 5
    for comp_num in range(max_comp):
        df[f"x{comp_num + 1}"] = df.cmp_mole_fractions.apply(
            lambda x: maybe_get_index(x, comp_num)
        )
        df[f"x{comp_num + 1}"].fillna(0, inplace=True)
        df[f"cmp{comp_num + 1}"] = df.cmp_ids.apply(
            lambda x: maybe_get_index(x, comp_num)
        )
        df[f"smi{comp_num + 1}"] = df[f"cmp{comp_num + 1}"].map(compound_lookup)
        df[f"smi{comp_num + 1}"].fillna("NONE", inplace=True)
    df["temperature"] = df["Temperature, K"]
    df_train = df.sample(frac=0.7, random_state=42)
    df_train.to_csv(os.path.join(output_dir, "train.csv"))
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.33, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"))
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"))


def process_olfactory_similarity(
    raw_datapath: str = "olfactory-similarity/processed_data/processed_OlfactorySimilarity.csv",
):
    raw_datapath = Path(raw_datapath)
    dataset_name = os.path.basename(raw_datapath.parent.parent)
    output_dir = os.path.join("chemixhub_mist", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_datapath.parent / "compounds.csv")
    compound_lookup = dict(zip(df.compound_id, df.smiles))
    df = pd.read_csv(raw_datapath)

    # Parse string representations of lists and map to SMILES
    df["mix1_smiles"] = df.cmp_ids_1.apply(
        lambda x: list(map(compound_lookup.get, eval(x)))
    )
    df["mix2_smiles"] = df.cmp_ids_2.apply(
        lambda x: list(map(compound_lookup.get, eval(x)))
    )
    df["target"] = df.value.values

    # Split into train/val/test
    df_train = df.sample(frac=0.7, random_state=42)
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df = df.drop(df_train.index)
    df_val = df.sample(frac=0.33, random_state=42)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test = df.drop(df_val.index)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    process_olfactory_similarity()
    process_drug_solubility()
    process_binary_mixtures()
    process_binary_mixtures(
        raw_datapath="logV/processed_data/processed_logV.csv",
    )
    process_miscible_solvents()
    process_MON()
    process_il_thermo(prop="Viscosity")
    process_il_thermo(prop="Electrical conductivity")
