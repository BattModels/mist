import pandas as pd


def process_prediction_with_ref(iter, targets):
    rows = []

    for row in iter:
        out = {
            "compounds": row["compounds"],
            "composition": row["composition"],
            "temperature": row["temperature"],
        }

        for idx, target in enumerate(targets):
            out[target] = row["y"][idx]
            out[f"excess {target}"] = row["y_excess"][idx]
            out[f"linear {target}"] = row["y_linear"][idx]
            out[f"relative excess {target}"] = row["y_excess"][idx] / (
                row["y_excess"][idx] + row["y_linear"][idx]
            )
        rows.append(out)

    return pd.DataFrame(rows)
