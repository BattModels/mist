import pandas as pd
import numpy as np
from electrolyte_fm.models.prod_finetune import MISTFinetuned

small_change = [
    ({"Acetone": "CC(=O)C"}, {"Thioacetone": "CC(=S)C"}),
    ({"Limonene": "CC1=CCC(CC1)C(=C)C"}, {"Eucalyptol": "CC1(C2CCC(O1)(CC2)C)C"}),
]

model = MISTFinetuned.from_pretrained("data/models/mist-26.9M-48kpooqf-odour")
model.eval()
_smi = ["CC(=O)C", "CC(=S)C", "CC1=CCC(CC1)C(=C)C", "CC1(C)OC2CCC1(C)CC2"]

pred = model.predict(_smi)
pred = {k: 0.1 * v["value"] for k, v in pred.items()}
df = pd.DataFrame(pred, index=_smi)

for i in range(9):
    pred = model.predict(_smi)
    pred = {k: 0.1 * v["value"] for k, v in pred.items()}
    df += pd.DataFrame(pred, index=_smi)

normalized_logits = 1 / (1 + np.exp(-1 * df))
binary_df = normalized_logits > 0.5
binary_df = binary_df.astype("int64")
normalized_logits.to_csv("small_change_normalized_logits.csv")

binary_df.to_csv("small_change.csv")
df


def get_non_zero_columns(row):
    return row[row != 0].index.tolist()


# Apply the function to each row
non_zero_cols_per_row = df.apply(get_non_zero_columns, axis=1)

# Print the results
for index, row in binary_df.iterrows():
    c = get_non_zero_columns(row)
    print(f"Row {index}: Non-zero columns: {c}")
