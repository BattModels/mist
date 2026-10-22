"""Render paper reference tables beside acquired validation contracts."""

import argparse
import csv
import json
from pathlib import Path

import yaml

ROW_END = "\\\\"


def latex(table: dict) -> str:
    columns = table["columns"]
    lines = [
        f"% {table['label']}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        "Model & " + " & ".join(columns) + " " + ROW_END,
        "\\midrule",
    ]
    for row in table["rows"]:
        cells = [
            f"{value:g} $\\pm$ {error:g}"
            for value, error in zip(row["values"], row["std"], strict=True)
        ]
        lines.append(row["model"] + " & " + " & ".join(cells) + " " + ROW_END)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def test_latex(rows: list[dict]) -> str:
    lines = [
        "% Computed held-out test scores; selection used validation data only.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Dataset & MAE & RMSE & AUROC & AP " + ROW_END,
        "\\midrule",
    ]
    for row in rows:
        scores = [
            f"{row['metrics'][metric]:.6g}"
            if row["metrics"].get(metric) is not None
            else "--"
            for metric in ("mae", "rmse", "auroc", "avg-precision")
        ]
        dataset = row["dataset"]
        if row.get("variant"):
            dataset += " (" + row["variant"] + ")"
        lines.append(" & ".join([row["base"], dataset, *scores]) + " " + ROW_END)
    return "\n".join([*lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text())
    acquired = [json.loads(path.read_text()) for path in args.tests]
    acquired.sort(
        key=lambda row: (row["base"], row["dataset"], row.get("variant") or "")
    )
    cases = {
        case["key"]: case for case in registry["cases"] if not case.get("target_subset")
    }
    expected = set(cases)
    actual = {row["case"] for row in acquired}
    if actual != expected or len(actual) != len(acquired):
        raise ValueError(
            f"Incomplete or duplicate benchmark results: missing={expected - actual}, extra={actual - expected}"
        )
    if any(row["split"] != "test" for row in acquired):
        raise ValueError("Benchmark report requires held-out test scores")
    for row in acquired:
        case = cases[row["case"]]
        if (
            row["base"] != case["base"]
            or row["dataset"] != case.get("canonical_task", case.get("dataset"))
            or row.get("variant") != case.get("variant")
        ):
            raise ValueError(f"{row['case']}: results do not match the registered case")
        if row.get("selected_by") != "validation only":
            raise ValueError(
                f"{row['case']}: missing validation-only selection evidence"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "selection_split": "validation",
        "selection_test_used": False,
        "held_out_test_scored": True,
        "paper_tables": registry["paper_tables"],
        "acquired_models": acquired,
        "held_out_tests": [json.loads(path.read_text()) for path in args.tests],
    }
    (args.output_dir / "validation-benchmarks.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "benchmark-tests.tex").write_text(test_latex(acquired))
    with (args.output_dir / "benchmark-tests.csv").open("w", newline="") as stream:
        flat = [
            {
                "case": row["case"],
                "base": row["base"],
                "dataset": row["dataset"],
                "variant": row.get("variant"),
                "split": row["split"],
                "provenance": row["provenance"],
                **row["metrics"],
            }
            for row in acquired
        ]
        metric_fields = sorted(
            {field for row in flat for field in row if field not in flat[0]}
        )
        writer = csv.DictWriter(stream, fieldnames=[*flat[0].keys(), *metric_fields])
        writer.writeheader()
        writer.writerows(flat)
    for name, table in registry["paper_tables"].items():
        (args.output_dir / f"{name}.tex").write_text(latex(table))
        with (args.output_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model", *table["columns"], "provenance"])
            for row in table["rows"]:
                writer.writerow([row["model"], *row["values"], row["provenance"]])


if __name__ == "__main__":
    main()
