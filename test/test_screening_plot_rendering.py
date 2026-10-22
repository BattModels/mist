"""Screening workflow rendering tests using synthetic fixtures."""

import csv
import itertools
import json
import math
import os
import random
import sqlite3
import subprocess
from pathlib import Path

import pytest

from test_workflow_plot_rendering import ROOT, run_plot, write_json

pytestmark = pytest.mark.skipif(
    os.environ.get("MIST_JULIA_SMOKE") != "1",
    reason="Opt-in Julia rendering integration test",
)


@pytest.mark.parametrize(
    ("base", "empty"), [("mist-28m", False), ("mist-1p8b", False), ("mist-1p8b", True)]
)
@pytest.mark.parametrize(
    ("analysis", "grid_rows", "count_key"),
    [("lithium-air", 4, "pareto_count"), ("electrolyte", 5, "pareto_molecules")],
)
def test_screening_structure_pages_match_pareto_ids_and_clean_stale_pages(
    tmp_path, base, empty, analysis, grid_rows, count_key
):
    from rdkit import Chem

    table = tmp_path / "pareto.csv"
    count = 0 if empty else grid_rows * 3 + 1
    smiles = ["C" * length for length in range(1, 12)] + [
        "N[C@@H](C)C(=O)O",
        "N[C@H](C)C(=O)O",
        "CCBr",
        "CCI",
        "COC",
    ]
    rows = [
        {
            "id": index,
            "smiles": smi,
            "inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles(smi)),
            "homo": -8.0 - index / 100,
        }
        for index, smi in enumerate(smiles[:count], start=1)
    ]
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "smiles", "inchi_key", "homo"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = tmp_path / "campaign.json"
    write_json(summary_path, {"base": base, "analysis": analysis, count_key: count})
    output = tmp_path / "fig"
    (output / "pages").mkdir(parents=True)
    (output / "molecules").mkdir()
    (output / "pages/grid_p99.pdf").write_text("old generated page")
    (output / "pages/notes.txt").write_text("user notes")
    (output / "molecules/mol_999.svg").write_text("old generated molecule")
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            analysis + "-structures",
            "--pareto",
            table,
            "--campaign-summary",
            summary_path,
            "--obabel",
            ROOT / "opt/screening/.venv/bin/obabel",
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["base"] == base
    assert report["analysis"] == analysis + "-structures"
    assert report["page_count"] == math.ceil(count / (grid_rows * 3))
    assert report["molecule_count"] == count
    assert report["grid_rows"] == grid_rows and report["grid_columns"] == 3
    assert [number for page in report["pages"] for number in page["ids"]] == list(
        range(1, count + 1)
    )
    assert [row["inchi_key"] for row in report["molecules"]] == [
        row["inchi_key"] for row in rows
    ]
    assert (output / "structures.csv").read_bytes() == table.read_bytes()
    assert not (output / "pages/grid_p99.pdf").exists()
    assert not (output / "molecules/mol_999.svg").exists()
    assert (output / "pages/notes.txt").read_text() == "user notes"
    for page in report["pages"]:
        for key in ["svg", "pdf", "png"]:
            assert (output / page[key]).stat().st_size > 100
        assert (output / page["pdf"]).read_bytes().startswith(b"%PDF")
    latex = (output / "structures.tex").read_text()
    assert latex.count(r"\begin{figure}") == report["page_count"]
    if count == 0:
        assert report["status"] == "no-passing-molecules"
        assert "no Pareto structure pages" in latex
        assert not list((output / "molecules").iterdir())
    else:
        assert "Page 2 of 2" in latex and "33" not in latex


@pytest.mark.parametrize(
    ("base", "empty"), [("mist-28m", False), ("mist-1p8b", False), ("mist-1p8b", True)]
)
def test_screening_lithium_air_renders_all_analyses_and_preserves_units(
    tmp_path, base, empty
):
    import numpy as np
    import yaml
    from rdkit import Chem

    from electrolyte_fm.interpretibility.functional_groups import (
        identify_functional_groups,
    )

    hartree_to_ev = 27.211386245981
    config = yaml.safe_load(
        (ROOT / "opt/screening/config_LiO2_battery.yaml").read_text()
    )
    heads = ["bp", "mp", "pka", "etn", "kt", "qm9"]
    for head, critic in zip(heads, config["critics"], strict=True):
        critic["model_path"] = f"/{base}/{head}.ckpt"
    config_path = tmp_path / "config.json"
    write_json(config_path, config)

    def molecule(smiles, homo, pka, beta, etn, mp, bp):
        return {
            "smi": smiles,
            "inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)),
            "homo": float(np.float32(homo)),
            "DMSO_pKa": pka,
            "beta": beta,
            "ETN": etn,
            "mp": mp,
            "bp": bp,
        }

    generated = [
        molecule(
            "CCOC", -0.32, 35, 0.6, 0.3, -150 if base == "mist-1p8b" else -50, 140
        ),
        molecule("CCN", -0.29, 32, 0.4, 0.25, -10, 100),
        molecule("CCF", -0.4, 38, 0.2, 0.4, -30, 120),
        molecule("CCC", -7.01 / hartree_to_ev, 30, 0.19, 0.46, 25, 75),
    ]

    def energy(row):
        # Independent implementation of the manuscript's two polynomials.
        beta, etn = row["beta"], row["ETN"]
        return (-3.330 - 0.226 * beta - 1.075 * beta**2) - (
            -0.948 + 0.192 * etn + 0.427 * etn**2
        )

    # A duplicate alias must not inflate reference counts. 'Fe' is not fluorine.
    references = [
        generated[0],
        {**generated[0], "smi": "COCC"},
        molecule("[Fe+2]", -0.22, 26, 0.9, 0.8, 35, 60),
    ]
    if empty:
        generated = []
    database = tmp_path / "merged.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE molecules (inchi TEXT PRIMARY KEY, smiles TEXT, props TEXT)"
        )
        for row in generated:
            properties = {
                key: value
                for key, value in row.items()
                if key not in {"smi", "inchi_key"}
            }
            properties["delta_G_HA_sol"] = float(np.float32(energy(row)))
            connection.execute(
                "INSERT INTO molecules VALUES (?, ?, ?)",
                (row["inchi_key"], row["smi"], json.dumps(properties)),
            )
    reference_path = tmp_path / "references.csv"
    with reference_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    metadata = tmp_path / "references.json"
    write_json(
        metadata,
        {
            "base": base,
            "analysis": "lithium-air",
            "rows": 3,
            "unique_molecules": 2,
            "selections": {
                head: {"checkpoint": f"/{base}/{head}.ckpt"} for head in heads
            },
            "units": {
                "homo": "hartree",
                "bp": "degC",
                "mp": "degC",
                "DMSO_pKa": "dimensionless",
                "ETN": "dimensionless",
                "beta": "dimensionless",
            },
        },
    )
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            "lithium-air",
            "--database",
            database,
            "--config",
            config_path,
            "--references",
            reference_path,
            "--reference-summary",
            metadata,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    for name in [
        "lio2-pareto",
        "lio2-homo-colored",
        "lio2-homo-pka",
        "lio2-pka-habstraction",
        "lio2-nucleophilic-attack",
        "lio2-functional-groups",
    ]:
        for extension in ["png", "pdf"]:
            assert (output / f"{name}.{extension}").stat().st_size > 100
    report = json.loads((output / "summary.json").read_text())
    assert report["base"] == base and report["generated"] == len(generated)
    assert report["reference_unique"] == 2 and report["reference_passing"] == 1
    assert report["novel_generated"] == (0 if empty else 3)
    assert report["novel_fraction"] == (None if empty else 0.75)
    assert report["status"] == ("no-passing-molecules" if empty else "ok")
    assert report["fluorinated"] == {"Generated": 0 if empty else 1, "Reference": 0}

    def read_csv(filename):
        with (output / filename).open() as stream:
            return list(csv.DictReader(stream))

    for filename, source in [
        ("generated.csv", generated),
        ("references.csv", [references[0], references[-1]]),
    ]:
        observed = {row["inchi_key"]: row for row in read_csv(filename)}
        for row in source:
            actual = observed[row["inchi_key"]]
            # Plotting all six figures must not repeatedly convert the inputs.
            assert float(actual["homo"]) == pytest.approx(row["homo"])
            assert float(actual["delta_G_HA_sol"]) == pytest.approx(energy(row))
            assert float(actual["rH"]) == pytest.approx(
                math.exp(-math.log(10) / 2 * row["DMSO_pKa"]), abs=0
            )
    counts = read_csv("functional-groups.csv")
    for row in counts:
        group = row["functional_group"]
        assert int(row["generated"]) == sum(
            group in identify_functional_groups(r["smi"]) for r in generated
        )
        assert int(row["reference"]) == sum(
            group in identify_functional_groups(r["smi"])
            for r in [references[0], references[-1]]
        )
    if empty:
        assert all(int(row["generated"]) == 0 for row in counts)
    else:
        assert any(int(row["reference"]) == 0 for row in counts)

    def objectives(row):
        return [row["mp"], -row["bp"], -row["DMSO_pKa"], energy(row), row["homo"]]

    def dominates(left, right):
        pairs = list(zip(objectives(left), objectives(right)))
        return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)

    expected_front = {
        row["inchi_key"]
        for row in generated
        if not any(dominates(other, row) for other in generated)
    }
    front = read_csv("pareto.csv")
    assert {row["inchi_key"] for row in front} == expected_front
    assert report["pareto_count"] == len(front)
    by_key = {row["inchi_key"]: row for row in generated}
    for row in front:
        assert float(row["homo"]) == pytest.approx(
            by_key[row["inchi_key"]]["homo"] * hartree_to_ev
        )
    assert (output / "pareto.smi").read_text().splitlines() == [
        row["smiles"] for row in front
    ]
    table = (output / "pareto.tex").read_text()
    assert (
        r"\begin{longtable}" in table and r"\endhead" in table and "HOMO (eV)" in table
    )


def test_screening_entropy_matches_legacy_complete_graph_mst():
    result = subprocess.run(
        [
            "uv",
            "run",
            "julia",
            "--project=plots",
            "--startup-file=no",
            "-e",
            """
        using ScreeningPlots, Graphs, Random, Test, Statistics, LinearAlgebra
        Random.seed!(17)
        for d in [2, 7, 512, 1536]
            E = randn(9, d)
            for distance in [ScreeningPlots.eculidean_distance, ScreeningPlots.cosine_distance, ScreeningPlots.angular_distance]
                dd = [distance(E[i, :], E[j, :]) for i in 1:9, j in 1:9]
                legacy = boruvka_mst(complete_graph(9), dd).weight
                @test ScreeningPlots.embedding_mst_distance(E; distance) ≈ legacy
                value = ScreeningPlots.renyi_entropy_estimate(E; distance)
                gamma = (d - 1) / d
                @test value.renyi ≈ (log(legacy) - gamma * log(9)) / (1 - gamma)
                @test value.status == "ok"
            end
            @test ScreeningPlots.embedding_momemt(E) ≈ mean(ScreeningPlots.embedding_spread(E))
            @test ScreeningPlots.bounding_diameter(E) ≈ norm(maximum(E; dims=1) - minimum(E; dims=1))
        end
        @test ismissing(ScreeningPlots.renyi_entropy_estimate(ones(1, 12)).renyi)
        empty = ScreeningPlots.renyi_entropy_estimate(zeros(0, 12))
        @test empty.status == "empty_cohort"
        @test ismissing(empty.mst) && ismissing(empty.renyi)
        @test_throws ErrorException ScreeningPlots.embedding_mst_distance(zeros(0, 12))
        @test ScreeningPlots.renyi_entropy_estimate(ones(3, 12)).status == "zero_mst_length"
        @test_throws ErrorException ScreeningPlots.renyi_entropy_estimate(fill(NaN, 3, 12))
        """,
        ],
        cwd=ROOT / "opt/screening",
        env={
            **os.environ,
            "UV_NO_SYNC": "1",
            "JULIA_PYTHONCALL_EXE": str(ROOT / "opt/screening/.venv/bin/python"),
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("base,dimension", [("mist-28m", 512), ("mist-1p8b", 1536)])
@pytest.mark.parametrize("empty_generated", [False, True])
def test_screening_creativity_renders_and_matches_independent_entropy(
    tmp_path, base, dimension, empty_generated
):
    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist

    rows = []
    for group in ["Generated", "Electrolytes", "ChEMBL"]:
        if empty_generated and group == "Generated":
            continue
        # One ChEMBL row explicitly exercises an undefined entropy, not a fake zero.
        for index in range(1 if group == "ChEMBL" else 6):
            rows.append(
                {
                    "group": group,
                    "inchi_key": f"{group}-{index}",
                    "smiles": "C",
                    "homo": -0.4 + 0.003 * index,
                    "lumo": -0.1 + 0.005 * index,
                    "gap": 0.3 + 0.002 * index,
                    "mp": -40.0 + index,
                    "bp": 200.0 + 3 * index,
                    "surprise": 10.0 + 3 * index + (5 if group == "Generated" else 0),
                    "tokens": 10,
                }
            )
    # Ensure the Pareto overlay has both on-front and dominated generated points.
    if not empty_generated:
        rows[5].update(homo=-0.2, gap=0.1, mp=50.0, bp=100.0)
    scores = tmp_path / "scores.csv"
    with scores.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "base": base,
        "rows": len(rows),
        "dimension": dimension,
        "cohort_summary": {
            "base": base,
            "counts": {
                group: {"unique_molecules": sum(row["group"] == group for row in rows)}
                for group in ["Generated", "Electrolytes", "ChEMBL"]
            },
        },
    }
    summary = tmp_path / "scores.json"
    write_json(summary, metadata)
    rng = np.random.default_rng(5)
    matrices = {
        pool: rng.normal(size=(len(rows), dimension)).astype(np.float32)
        for pool in ["first", "mean_masked"]
    }
    embeddings = tmp_path / "embeddings.npz"
    np.savez(
        embeddings,
        **matrices,
        group=[row["group"] for row in rows],
        inchi_key=[row["inchi_key"] for row in rows],
    )
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--analysis",
            "creativity",
            "--base",
            base,
            "--scores",
            scores,
            "--score-summary",
            summary,
            "--embeddings",
            embeddings,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    for name in [
        "surprise-vs-utility.pdf",
        "surprise-vs-utility.png",
        "entropy.csv",
        "entropy.tex",
        "creativity.csv",
    ]:
        assert (output / name).stat().st_size > 100
    report = json.loads((output / "summary.json").read_text())
    assert report["base"] == base and report["inference"] == metadata
    assert report["status"] == ("no-passing-molecules" if empty_generated else "ok")
    assert len(report["metrics"]) == 18
    for value in report["metrics"]:
        if value["group"] == "Generated" and empty_generated:
            assert value["n"] == 0 and value["status"] == "empty_cohort"
            assert value["mean_surprise"] is None
            assert value["mst"] is None and value["renyi"] is None
            continue
        if value["group"] == "ChEMBL":
            assert value["renyi"] is None and value["mst"] == 0
            assert value["status"] == "insufficient_samples"
            continue
        indices = [i for i, row in enumerate(rows) if row["group"] == value["group"]]
        E = matrices[value["pool"]][indices].astype(np.float64)
        if value["metric"] == "euclidean":
            distances = cdist(E, E)
        else:
            normed = E / np.linalg.norm(E, axis=1, keepdims=True)
            cosine = normed @ normed.T
            distances = (
                np.sqrt(2 * np.maximum(1 - cosine, 0))
                if value["metric"] == "cosine"
                else np.arccos(np.clip(cosine, 0, 1))
            )
        np.fill_diagonal(distances, 0)
        length = minimum_spanning_tree(distances).sum()
        gamma = (dimension - 1) / dimension
        expected = (np.log(length) - gamma * np.log(len(indices))) / (1 - gamma)
        assert value["mst"] == pytest.approx(length)
        assert value["renyi"] == pytest.approx(expected)
        assert value["mean_surprise"] == pytest.approx(
            np.mean([rows[i]["surprise"] for i in indices])
        )
    assert report["cohorts"]["Generated"]["pareto_count"] == (
        0 if empty_generated else 5
    )
    if empty_generated:
        assert report["cohorts"]["Generated"]["mean_surprise"] is None
    assert report["cohorts"]["Electrolytes"]["dominated_mean_surprise"] is None
    with (output / "creativity.csv").open() as stream:
        returned = list(csv.DictReader(stream))
    assert [float(row["homo"]) for row in returned] == [row["homo"] for row in rows]
    table = (output / "entropy.tex").read_text()
    assert "\\hat{H}_{\\gamma}" in table and "ChEMBL & 1 & 0.0 & --" in table
    if empty_generated:
        assert "Generated & 0 & -- & -- & --" in table


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
def test_screening_performance_uses_all_array_cells_and_generation_time(tmp_path, base):
    configs = []
    rates = []
    # Deliberately unlike the historical 88.8M / 122 inventory sizes.
    inventory_count = 1_000_000 if base == "mist-28m" else 2_000_000
    reference_count = 173
    for index, (gpus, batch, fragments, epoch, limit) in enumerate(
        itertools.product(
            (1, 2, 4, 8),
            (128, 256, 512, 1024),
            (4000, 8000),
            (0.9, 1.0, 1.1),
            (None, 0.5, 0.25, 0.1),
        )
    ):
        run = tmp_path / f"{base}-{index}"
        config = run / "config.json"
        write_json(
            config,
            {
                "limit_walltime": 300,
                "generation": {
                    "batch_size": batch,
                    "n_fragments": fragments,
                    "epoch_size": int(epoch * fragments),
                    "limit_db_fragments": None
                    if limit is None
                    else int(limit * inventory_count),
                    "limit_ref_fragments": None,
                    "inventory": {
                        "database": inventory_count,
                        "reference": reference_count,
                    },
                },
            },
        )
        throughput = (
            2000
            * gpus**0.8
            * math.exp(
                0.0001 * batch
                - 0.00000002 * batch**2
                + 0.00002 * fragments
                + 0.1 * epoch
            )
        )
        efficiency = (
            0.8
            + 0.01 * math.log(gpus)
            + 0.1 * epoch
            - 0.05 * epoch**2
            - 0.01 * (limit or 1)
        )
        messages = [
            # The prior attempt must not leak its ninth GPU or timing into this attempt.
            {"global_rank": 8, "passing_rank": 1, "eval_throughput_rank": 999999},
            {
                "global_rank": 0,
                "n_passing_world": 999,
                "unique_molecules_world": 999999,
                "elapsed_perf": 1000,
            },
            {"global_rank": 0, "screening_start": True, "elapsed_perf": 600},
        ]
        for rank in range(gpus):
            messages.append(
                {
                    "global_rank": rank,
                    "passing_rank": 9,
                    "eval_throughput_rank": throughput / gpus,
                    "generation_elapsed": 300,
                    "final": True,
                }
            )
        for seconds in (60, 120, 180, 240, 300):
            messages.append(
                {
                    "global_rank": 0,
                    "elapsed_perf": 600 + seconds,
                    "generation_elapsed": seconds,
                    "n_passing_world": seconds,
                    "unique_molecules_world": int(throughput * seconds * efficiency),
                    "final": seconds == 300,
                }
            )
        (run / "screen.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in messages)
        )
        configs.append(config)
        rates.append(throughput)
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            "performance",
            "--configs",
            *configs,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    for filename in (
        "scaling.pdf",
        "scaling.png",
        "screening_lm.tex",
        "performance.csv",
    ):
        assert (output / filename).stat().st_size > 100
    report = json.loads((output / "summary.json").read_text())
    assert report["base"] == base and report["runs"] == 384
    assert report["source_configs"] == list(map(str, configs))
    assert report["inventory_counts"] == {
        "database": inventory_count,
        "reference": reference_count,
    }
    with (output / "performance.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 384
    assert {float(row["duration"]) for row in rows} == {300.0}
    assert [float(row["global_throughput"]) for row in rows] == pytest.approx(rates)
    assert {int(row["gpus"]) for row in rows} == {1, 2, 4, 8}
    assert {float(row["limit_db_fragments"]) for row in rows} == {1, 0.5, 0.25, 0.1}
    assert {float(row["limit_ref_fragments"]) for row in rows} == {1}
    assert report["efficiency"]["coefficients"]["limit_db_fragments"] == pytest.approx(
        -0.01, abs=1e-6
    )


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize("size", [12, 0])
def test_screening_qmist_parity_reads_workflow_jsonl_and_sqlite(tmp_path, base, size):
    database = tmp_path / "merged.sqlite"
    measured = []
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE molecules (inchi TEXT PRIMARY KEY, smiles TEXT, props TEXT)"
        )
        for index in range(size):
            values = {
                "gap": 0.2 + 0.005 * index,
                "homo": -0.32 + 0.004 * index,
                "lumo": -0.12 + 0.009 * index,
            }
            # Vary property order: names, not dictionary order, define columns.
            prediction = {
                key: values[key] + 0.001
                for key in sorted(values, reverse=index % 2 == 0)
            }
            connection.execute(
                "INSERT INTO molecules VALUES (?, ?, ?)",
                (f"key-{index}", "C" * (index + 1), json.dumps(prediction)),
            )
            measured.append({"InChIKey": f"key-{index}", **values})
    if size == 0 and base == "mist-1p8b":
        # The root workflow combines every family/campaign into one QMiST file.
        # Results from another campaign do not create pairs for this empty one.
        measured.append(
            {"InChIKey": "other-campaign", "homo": -0.3, "lumo": 0.1, "gap": 0.4}
        )
    qmist = tmp_path / "results.jsonl"
    qmist.write_text("".join(json.dumps(row) + "\n" for row in measured))
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            "qmist",
            "--database",
            database,
            "--qmist",
            qmist,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    for filename in ("qmist-parity.pdf", "qmist-parity.png"):
        assert (output / filename).stat().st_size > 100
    summary = json.loads((output / "summary.json").read_text())
    assert summary["base"] == base and summary["molecules"] == size
    assert summary["status"] == ("ok" if size else "no-passing-molecules")
    with (output / "qmist-parity.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert "gap_qmist" in reader.fieldnames and "homo_mist" in reader.fieldnames
        assert len(list(reader)) == size
    for metrics in summary["metrics"].values():
        assert metrics == (
            pytest.approx({"mae": 0.001, "rmse": 0.001})
            if size
            else {"mae": None, "rmse": None}
        )


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize("empty_generated", [False, True])
def test_screening_campaign_renders_pareto_trace_and_panel(
    tmp_path, base, empty_generated
):
    from rdkit import Chem

    molecules = {"a": "CCO", "b": "CCN", "c": "COC", "d": "C(=O)O", "e": "O"}
    identities = {
        label: Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        for label, smi in molecules.items()
    }
    database = tmp_path / "merged.sqlite"
    values = [
        ("a", -0.6, 0.6, -100.0, 300.0),
        ("b", -0.5, 0.5, -80.0, 250.0),
        ("c", -0.45, 0.7, -120.0, 280.0),
        ("d", -0.3, 0.2, 0.0, 75.0),
    ]
    if empty_generated:
        values = []
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE molecules (inchi TEXT PRIMARY KEY, smiles TEXT, props TEXT)"
        )
        for key, homo, gap, mp, bp in values:
            connection.execute(
                "INSERT INTO molecules VALUES (?, ?, ?)",
                (
                    identities[key],
                    molecules[key],
                    json.dumps(
                        {
                            "homo": homo,
                            "lumo": homo + gap,
                            "gap": gap,
                            "mp": mp,
                            "bp": bp,
                        }
                    ),
                ),
            )
    references = tmp_path / "references.csv"
    references.write_text(
        "inchi_key,homo,lumo,gap,mp,bp\n"
        f"{identities['a']},-0.6,0.0,0.6,-100,300\n"
        f"{identities['e']},-0.3,0.1,0.4,-10,200\n"
        f"{identities['e']},-0.3,0.1,0.4,-10,200\n"
    )
    reference_summary = tmp_path / "references.json"
    config = tmp_path / "config.json"
    write_json(
        config,
        {"critics": [{"model_path": f"/{head}.ckpt"} for head in ["qm9", "bp", "mp"]]},
    )
    write_json(
        reference_summary,
        {
            "base": base,
            "selections": {
                head: {"checkpoint": f"/{head}.ckpt"} for head in ["qm9", "bp", "mp"]
            },
            "units": {
                "homo": "hartree",
                "lumo": "hartree",
                "gap": "hartree",
                "mp": "degC",
                "bp": "degC",
            },
        },
    )
    performance = tmp_path / "performance.csv"
    performance.write_text(
        "gpus,batch_size,global_throughput\n1,128,10000\n2,256,19000\n4,512,35000\n8,1024,70000\n"
    )
    performance_summary = tmp_path / "performance.json"
    write_json(performance_summary, {"base": base})
    log = tmp_path / "screen.jsonl"
    # A single final snapshot is enough to draw a valid trace. The other family
    # exercises the full trace and counts beyond the legacy hard-coded limits.
    points = [8000] if base == "mist-1p8b" else [1000, 3000, 5000, 6000, 8000]
    log.write_text(
        "".join(
            json.dumps(
                {
                    "global_rank": 0,
                    "elapsed_perf": 600 + 100 * index,
                    "generation_elapsed": 100 * index,
                    "n_passing_world": 0 if empty_generated else passing,
                    "unique_molecules_world": 60_000_000 * index,
                }
            )
            + "\n"
            for index, passing in enumerate(points, start=1)
        )
    )
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            "electrolyte",
            "--database",
            database,
            "--config",
            config,
            "--references",
            references,
            "--reference-summary",
            reference_summary,
            "--performance",
            performance,
            "--performance-summary",
            performance_summary,
            "--log",
            log,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    for name in ["pareto", "generation-trace", "screening-panel"]:
        for extension in ["pdf", "png"]:
            assert (output / f"{name}.{extension}").stat().st_size > 100
    summary = json.loads((output / "summary.json").read_text())
    assert summary["base"] == base and summary["pareto_molecules"] == (
        0 if empty_generated else 2
    )
    assert summary["status"] == ("no-passing-molecules" if empty_generated else "ok")
    assert (
        summary["generated_molecules"] == len(values)
        and summary["reference_molecules"] == 2
    )
    assert summary["novel_molecules"] == (0 if empty_generated else 3) and summary[
        "reference_molecules_not_generated"
    ] == (2 if empty_generated else 1)
    assert summary["passing_estimate"] == (0 if empty_generated else 8000)
    with (output / "pareto-front.csv").open() as stream:
        front = {row["inchi_key"]: row for row in csv.DictReader(stream)}
    assert set(front) == (
        set() if empty_generated else {identities["a"], identities["c"]}
    )
    if not empty_generated:
        assert float(front[identities["a"]]["homo"]) == pytest.approx(
            -0.6 * 27.211386245981
        )
        assert float(front[identities["c"]]["thermal_window"]) == 400
    table = (output / "pareto-front.tex").read_text()
    assert r"\begin{longtable}" in table and r"\endhead" in table
    assert "HOMO (eV)" in table
    if not empty_generated:
        assert "-16.3" in table

    # Exercise the actual output contract between the two workflow tasks,
    # including the header-only frontier from a zero-yield campaign.
    structures = tmp_path / "structures"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--analysis",
            "electrolyte-structures",
            "--base",
            base,
            "--pareto",
            output / "pareto-front.csv",
            "--campaign-summary",
            output / "summary.json",
            "--obabel",
            ROOT / "opt/screening/.venv/bin/obabel",
            "--output-dir",
            structures,
        ],
        "opt/screening",
    )
    pages = json.loads((structures / "summary.json").read_text())
    assert pages["molecule_count"] == summary["pareto_molecules"]
    assert pages["grid_rows"] == 5 and pages["grid_columns"] == 3
    assert pages["page_count"] == (0 if empty_generated else 1)
    assert [row["inchi_key"] for row in pages["molecules"]] == list(front)
    assert (structures / "structures.csv").read_bytes() == (
        output / "pareto-front.csv"
    ).read_bytes()


def test_screening_qmist_validation_replays_each_published_version(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    reference = [
        {
            "smiles": "C" * (i + 1),
            "homo": -0.3 + 0.01 * i,
            "lumo": 0.01 * i,
            "gap": 0.3 + 0.01 * i,
        }
        for i in range(4)
    ]
    (release / "qm9.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reference)
    )
    for version, error in [("v1", 0.01), ("v2", 0.001), ("v3", 0.0001)]:
        for index, row in enumerate(
            [*reference, {"smiles": "O", "homo": -0.4, "lumo": 0.1, "gap": 0.5}]
        ):
            write_json(
                release / f"veri_{version}" / f"{index}.json",
                {
                    "smiles": row["smiles"],
                    "InChIKey": f"key-{index}",
                    "walltime": 100.0 + index,
                    **{key: row[key] + error for key in ["homo", "lumo", "gap"]},
                },
            )
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--analysis",
            "qmist-validation",
            "--release",
            release,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["mode"] == "published-results-replay"
    assert summary["validates_current_qmist"] is False
    assert summary["reference_rows"] == 4
    for version, error in [("v1", 0.01), ("v2", 0.001), ("v3", 0.0001)]:
        report = summary["versions"][version]
        assert report["source_rows"] == 5 and report["matched_rows"] == 4
        assert report["unmatched_source_smiles"] == 1
        for metrics in report["metrics"].values():
            assert metrics == pytest.approx({"mae": error, "rmse": error})
        for extension in ["png", "pdf", "csv"]:
            assert (output / f"{version}-parity.{extension}").stat().st_size > 100


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize("analysis", ["olfaction", "qmist-chembl"])
@pytest.mark.parametrize("empty_generated", [False, True])
def test_screening_comparison_plots_use_all_cohorts_and_named_channels(
    tmp_path, base, analysis, empty_generated
):
    rows = []
    quantum = []
    channels = ["odorless", "floral", "fruity"]
    counts = {}
    for group, size in [("Generated", 4), ("Electrolytes", 2), ("ChEMBL", 3)]:
        if empty_generated and group == "Generated":
            size = 0
        counts[group] = {"source_rows": size, "unique_molecules": size}
        for i in range(size):
            row = {
                "group": group,
                "inchi_key": f"{group}-{i}",
                "smiles": "C" * (i + 1),
                "homo": -0.6 + 0.05 * i,
                "lumo": 0.01 * i,
                "gap": 0.6 - 0.04 * i,
                "mp": -100 + 10 * i,
                "bp": 300 - 10 * i,
                "odorless": [-1, 1, 0, 2][i],
                "floral": [-1, 1, -1, 1][i],
                "fruity": 1.0,
            }
            if group == "Electrolytes":
                row["odorless"] = -1.0
            if base == "mist-1p8b":
                row.update({channel: -1.0 for channel in channels})
            if group == "ChEMBL":
                row.update(
                    {
                        col + "_qmist": row[col] + 0.002
                        for col in ["homo", "lumo", "gap"]
                    }
                )
            elif group == "Generated":
                quantum.append(
                    {
                        "InChIKey": row["inchi_key"],
                        **{col: row[col] + 0.001 for col in ["homo", "lumo", "gap"]},
                    }
                )
            rows.append(row)
    data = tmp_path / "cohorts.csv"
    with data.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = tmp_path / "cohort-summary.json"
    write_json(metadata, {"base": base, "counts": counts, "odour_channels": channels})
    if empty_generated and base == "mist-1p8b":
        quantum.append(
            {"InChIKey": "other-campaign", "homo": -0.3, "lumo": 0.1, "gap": 0.4}
        )
    qmist = tmp_path / "qmist.jsonl"
    qmist.write_text("".join(json.dumps(row) + "\n" for row in quantum))
    output = tmp_path / "fig"
    run_plot(
        "opt/screening/plots/workflow.jl",
        [
            "--base",
            base,
            "--analysis",
            analysis,
            "--cohorts",
            data,
            "--cohort-summary",
            metadata,
            "--qmist",
            qmist,
            "--output-dir",
            output,
        ],
        "opt/screening",
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["base"] == base and summary["analysis"] == analysis
    assert summary["status"] == ("no-passing-molecules" if empty_generated else "ok")
    if analysis == "olfaction":
        names = ["screening-odors", "electrolyte-odorless", "screening-olfaction"]
        if base == "mist-1p8b":
            assert summary["positive_channels"] == 0
            assert all(
                value == 0
                for group in summary["active_fraction"].values()
                for value in group.values()
                if value is not None
            )
        else:
            assert summary["positive_channels"] == (2 if empty_generated else 3)
            if not empty_generated:
                assert summary["active_fraction"]["Generated"] == {
                    "odorless": 0.5,
                    "floral": 0.5,
                    "fruity": 1.0,
                }
        assert (output / "odor-counts.csv").is_file()
        if empty_generated:
            assert summary["active_fraction"]["Generated"] == dict.fromkeys(channels)
            with (output / "odor-counts.csv").open() as stream:
                odors = list(csv.DictReader(stream))
            generated = [row for row in odors if row["group"] == "Generated"]
            assert len(generated) == summary["positive_channels"]
            assert all(
                row["cohort_size"] == "0" and row["value"] == "" for row in generated
            )
    else:
        names = ["qmist-chembl-parity"]
        n_generated = 0 if empty_generated else 4
        for group, size, error in [
            ("Generated", n_generated, 0.001),
            ("ChEMBL", 3, 0.002),
        ]:
            assert summary["metrics"][group]["molecules"] == size
            for metrics in summary["metrics"][group]["properties"].values():
                assert metrics == (
                    pytest.approx({"mae": error, "rmse": error})
                    if size
                    else {"mae": None, "rmse": None}
                )
        with (output / "qmist-chembl-parity.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == n_generated + 3
        for metrics in summary["metrics"]["Combined"]["properties"].values():
            assert metrics == pytest.approx(
                {
                    "mae": (n_generated * 0.001 + 3 * 0.002) / (n_generated + 3),
                    "rmse": math.sqrt(
                        (n_generated * 0.001**2 + 3 * 0.002**2) / (n_generated + 3)
                    ),
                }
            )
    for name in names:
        for extension in ["pdf", "png"]:
            assert (output / f"{name}.{extension}").stat().st_size > 100
