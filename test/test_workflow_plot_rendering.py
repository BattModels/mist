"""Opt-in render tests: MIST_JULIA_SMOKE=1 pytest test/test_workflow_plot_rendering.py.

These use synthetic fixtures, not scientific results or production checkpoints.
Instantiate each workflow's Julia environment before running them.
"""

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

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("MIST_JULIA_SMOKE") != "1",
    reason="Opt-in Julia rendering integration test",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run_plot(script, args, folder):
    env = {
        **os.environ,
        "UV_NO_SYNC": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    python = ROOT / folder / ".venv/bin/python"
    if python.is_file():
        env["JULIA_PYTHONCALL_EXE"] = str(python)
    result = subprocess.run(
        [str(ROOT / script), *map(str, args)],
        cwd=ROOT / folder,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode:
        output = Path(args[args.index("--output-dir") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / "plot.log"
        log.write_text(result.stdout + result.stderr)
        lines = result.stderr.splitlines()
        context = "\n".join(lines[:15] + lines[-60:])
        pytest.fail(f"{script} exited {result.returncode}; full log: {log}\n{context}")


def test_synth_access_aligns_rows_without_changing_the_cohort():
    subprocess.run(
        [
            "julia",
            "--project=.",
            "--startup-file=no",
            "-e",
            """
            module ScoresUnderTest
                include("plot.jl")
            end
            using DataFrames, Test
            align = ScoresUnderTest.align_scores
            left = DataFrame(row_id=[0, 1, 2], smiles=["CCO", "CCN", "CC"],
                target=[false, true, false], model=[.1, .8, .2])
            right = DataFrame(row_id=[2, 0, 1], smiles=["CC", "CCO", "CCN"],
                target=[false, false, true], heuristic=Any[2., nothing, 3.])
            joined = align(left, right, "fixture")
            @test joined.row_id == left.row_id
            @test joined.model == left.model
            @test isequal(joined.heuristic, [nothing, 3., 2.])
            @test_throws ErrorException align(left, right[1:2, :], "missing row")
            duplicate = vcat(right, right[1:1, :])
            @test_throws ErrorException align(left, duplicate, "duplicate")
            conflict = copy(right)
            conflict.target[1] = true
            @test_throws ErrorException align(left, conflict, "changed label")
            conflict = copy(right)
            conflict.smiles[1] = "CO"
            @test_throws ErrorException align(left, conflict, "changed molecule")
            @test_throws ErrorException align(left, select(right, Not(:row_id)), "missing ID")
            @test_throws ErrorException align(left[1:0, :], right[1:0, :], "empty")
            """,
        ],
        cwd=ROOT / "opt/synth_access",
        check=True,
        timeout=180,
    )


def test_synth_access_renders_all_figures(tmp_path):
    rng = random.Random(42)
    models = [
        "mist/mist-28m",
        "mist/mist-1p8b",
        "ibm-research/MoLFormer-XL-both-10pct",
        "seyonec/ChemBERTa-zinc-base-v1",
    ]
    heuristic_names = [
        "SAScore",
        "SCScore",
        "SyBA",
        "BR-SAScore",
        "smirk",
        "smirk-kekule",
        "smirk-canonical",
        "molecular-weight",
        "assembly-index",
    ]
    for dataset in ("crowdsourced", "ba-sascorer"):
        metadata = {
            "row_id": list(range(40)),
            "smiles": [f"molecule-{i}" for i in range(40)],
        }
        if dataset == "crowdsourced":
            metadata.update(
                meanComplexity=[rng.random() * 4 for _ in range(40)],
                stdevComplexity=[0.5] * 40,
                is_complex=[i % 2 == 0 for i in range(40)],
            )
        else:
            metadata["is_hard"] = [i % 2 == 0 for i in range(40)]
        auroc = {"meanComplexity": 0.8} if dataset == "crowdsourced" else {}
        scores = {**metadata}
        for metric in heuristic_names:
            scores[metric] = [rng.random() * 10 for _ in range(40)]
            auroc[metric] = 0.55 + rng.random() * 0.35
        # A native assembly timeout must survive JSON export as null and be
        # excluded pairwise from plots, not abort the entire report.
        scores["assembly-index"][0] = None
        write_json(
            tmp_path / "heuristics" / dataset / "scores.json",
            {"scores": scores, "auroc": auroc},
        )
        for model_index, model in enumerate(models):
            scores, auroc = {**metadata}, {}
            for untrained, encoding, per_token in itertools.product(
                (False, True),
                ("smiles", "smiles-kekule", "smiles-canonical"),
                (False, True),
            ):
                key = (
                    model
                    + ("-untrained" if untrained else "")
                    + "-"
                    + encoding
                    + ("-per-token" if per_token else "")
                )
                scores[key] = [rng.random() * 10 for _ in range(40)]
                auroc[key] = 0.55 + rng.random() * 0.35
            if model_index == 1:
                scores = {key: list(reversed(values)) for key, values in scores.items()}
            write_json(
                tmp_path / f"model-{model_index}" / f"{dataset}.json",
                {"scores": scores, "auroc": auroc},
            )
    output = tmp_path / "fig"
    run_plot(
        "opt/synth_access/plot.jl",
        [
            "--base",
            "mist-28m",
            "--model-dir",
            tmp_path / "model-0",
            "--base",
            "mist-1p8b",
            "--model-dir",
            tmp_path / "model-1",
            "--heuristics-dir",
            tmp_path / "heuristics",
            "--baseline-dir",
            tmp_path / "model-2",
            "--baseline-dir",
            tmp_path / "model-3",
            "--output-dir",
            output,
        ],
        "opt/synth_access",
    )
    for extension in ("pdf", "png"):
        figures = list(output.glob(f"*.{extension}"))
        assert len(figures) == 16
        assert all(path.stat().st_size > 100 for path in figures)
    assert (output / "interp_surprise.tex").is_file()


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
def test_feature_miner_renders_all_figures(tmp_path, base):
    import torch

    rng = random.Random(42)
    paths = []
    for dataset in ("pretrained", "tmQM", "QM9"):
        path = tmp_path / dataset / "checkpoints"
        path.mkdir(parents=True)
        paths.append(path)
        names = [
            f"*encoder.layer.{layer}.{location}"
            for location, layer in itertools.product(
                ("output", "intermediate", "attention", "output.dense"), range(4)
            )
        ]
        state = {"_prob_points": names}
        metrics = {}
        for index, name in enumerate(names):
            state[f"_probes.{index}.weight"] = torch.tensor(
                [[rng.random() for _ in range(8)] for _ in range(5)]
            )
            state[f"_probes.{index}.bias"] = torch.zeros(5)
            metrics[f"val/{name}-loss"] = 0.1 + rng.random() * 0.2
            metrics[f"val/{name}-auroc"] = 0.91 + rng.random() * 0.075
        for name in names:
            prefix = name.replace(".", "-").replace("*", "star")
            torch.save(
                {
                    "state_dict": state,
                    "hyper_parameters": {},
                    "epoch": 0,
                    "global_step": 8000,
                    "probe_metrics": metrics,
                },
                path / f"{prefix}--epoch-0--loss-0.100--auroc-0.950.ckpt",
            )
        torch.save(
            {
                "training_steps": 10_000 if base == "mist-28m" else 9000,
                "requested_steps": 10_000,
                "step_budget_reached": base == "mist-28m",
            },
            path.parent / "fit.pt",
        )
    output = tmp_path / "fig"
    run_plot(
        "opt/FeatureMiner/plots/lipinski_probes.jl",
        ["--base", base, "--output-dir", output, "--probes", *paths],
        "opt/FeatureMiner",
    )
    for extension in ("pdf", "png"):
        figures = list(output.glob(f"*.{extension}"))
        assert len(figures) == 5
        assert all(path.stat().st_size > 100 for path in figures)
    assert (output / "lipinski.tex").is_file()
    assert (output / "probes.jld2").is_file()
    # Existing JLD2 tables remain valid plot inputs; no checkpoint conversion
    # or JSON intermediate is required when re-rendering them.
    run_plot(
        "opt/FeatureMiner/plots/lipinski_probes.jl",
        [
            "--base",
            base,
            "--output-dir",
            tmp_path / "replot",
            "--probes",
            output / "probes.jld2",
        ],
        "opt/FeatureMiner",
    )
    assert (tmp_path / "replot/lipinski_linear_probes.pdf").is_file()


def test_design_theme_pipeline_preserves_theme_and_save_options(tmp_path):
    result = subprocess.run(
        [
            "uv",
            "run",
            "julia",
            "--project=.",
            "--startup-file=no",
            "-e",
            """
            using Makie, MISTStyle, Test
            output = ARGS[1]
            captured = Ref{Any}()
            with_theme(Theme(fontsize=42, Axis=(xlabelsize=31,))) do
                fig = with_theme(MISTStyle.theme()) do
                    themed = Figure()
                    axis = Axis(themed[1, 1])
                    @test axis.xlabelsize[] == MISTStyle.theme()[:Axis][:xlabelsize][]
                    @test Makie.current_default_theme()[:fontsize][] == 8
                    for style in (:regular, :bold)
                        font = Makie.to_font(themed.scene.theme[:fonts][style][])
                        @test font.units_per_EM > 0
                    end
                    bars = hist!(axis, [1., 2., 2., 3.])
                    @test to_color(bars.color[]) == to_color(MISTStyle.CAT_COLORS[1])
                    captured[] = (themed, axis)
                    themed
                end |> MISTStyle.savefig("themed"; fig_dir=output, dpi=72)
                @test fig === captured[][1]
                @test Makie.current_default_theme()[:fontsize][] == 42
                @test captured[][2].xlabelsize[] == 6
                @test isfile(joinpath(output, "themed.png"))
                @test isfile(joinpath(output, "themed.pdf"))
                overridden = joinpath(output, "override")
                unused = joinpath(output, "unused")
                saver = MISTStyle.savefig("custom"; fig_dir=unused, dpi=72)
                @test saver(fig; fig_dir=overridden, dpi=36) === fig
                @test isfile(joinpath(overridden, "custom.png"))
                @test isfile(joinpath(overridden, "custom.pdf"))
                @test !ispath(unused)
            end
            """,
            str(tmp_path),
        ],
        cwd=ROOT / "opt/design",
        env={
            **os.environ,
            "UV_NO_SYNC": "1",
            "JULIA_PYTHONCALL_EXE": str(ROOT / "opt/design/.venv/bin/python"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_design_error_lines_preserve_group_colors():
    # Checking that images exist did not detect the old recipe silently drawing
    # every group in black. Inspect both the line and its uncertainty bars.
    result = subprocess.run(
        [
            "uv",
            "run",
            "julia",
            "--project=.",
            "--startup-file=no",
            "-e",
            """
            using DesignRules, Makie, MISTStyle, Test
            with_theme(MISTStyle.theme()) do
                fig = Figure()
                axis = Axis(fig[1, 1])
                line = DesignRules.errorlines!(axis, [1, 2], [1., 2.]; color=:red)
                @test length(line.plots) == 2
                for child in line.plots
                    @test Makie.to_color(child[:color][]) == Makie.to_color(:red)
                end
                groups = DesignRules.errorlines!(axis, [1, 2], [1., 2.];
                    color=[1., 2.], colormap=:tab10, colorrange=(1, 10))
                for child in groups.plots
                    @test child[:color][] == [1., 2.]
                    @test Tuple(child[:colorrange][]) == (1, 10)
                end
            end
            """,
        ],
        cwd=ROOT / "opt/design",
        env={
            **os.environ,
            "UV_NO_SYNC": "1",
            "JULIA_PYTHONCALL_EXE": str(ROOT / "opt/design/.venv/bin/python"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("base", ["mist-28m", "mist-1p8b"])
@pytest.mark.parametrize(
    "analysis,expected_figures",
    [("hydrocarbons", 8), ("electrolytes", 1), ("permutations", 2), ("olfaction", 2)],
)
def test_design_renders_with_model_directories(
    tmp_path, monkeypatch, base, analysis, expected_figures
):
    import lightning.pytorch as pl
    import torch
    from smirk import SmirkTokenizerFast
    from transformers import RobertaConfig, RobertaModel

    from electrolyte_fm.models.lm_finetuning import LMFinetuning
    from electrolyte_fm.models.prod_finetune import MISTFinetuned
    from opt.benchmarks.scripts.partial_charge import build_model, predict_rows

    torch.set_num_threads(1)
    snapshot = tmp_path / "snapshot"
    tokenizer = SmirkTokenizerFast(template="[CLS] $0 [SEP]")
    tokenizer.save_pretrained(snapshot)
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=256,
    )
    RobertaModel(config, add_pooling_layer=False).save_pretrained(snapshot)
    qm = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "u0",
        "u298",
        "h298",
        "g298",
        "cv",
    ]
    heads = {"baseline": qm, "randomized": qm, "mp": ["mp"], "bp": ["bp"], "fp": ["fp"]}
    if analysis == "electrolytes":
        heads.update(kt=["alpha", "beta"], pka=["DMSO_pKa"], dn=["BF3 affinity"])
    elif analysis == "olfaction":
        heads = {"odour": ["floral", "woody", "fruity"]}
    args = []
    for name, channels in heads.items():
        model = LMFinetuning(
            output_size=len(channels),
            encoder_ckpt=str(snapshot),
            tokenizer=str(snapshot),
            task="binary" if name == "odour" else "regression",
            target_columns=channels,
            metrics=["auroc"] if name == "odour" else ["mae"],
            transform="identity",
            track_oov=False,
        )
        model.configure_model()
        checkpoint = tmp_path / f"{name}.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            checkpoint,
        )
        model_dir = tmp_path / name
        MISTFinetuned.from_lightning_checkpoint(
            checkpoint, tokenizer=tokenizer, local_files_only=True
        ).save_pretrained(model_dir)
        args.extend(["--model", name, model_dir])
    if analysis == "electrolytes":
        import csv

        from rdkit import Chem

        with (ROOT / "opt/design/electrolytes.csv").open() as stream:
            smiles = list(
                dict.fromkeys(
                    Chem.MolToSmiles(Chem.MolFromSmiles(r["smi"]))
                    for r in csv.DictReader(stream)
                )
            )
        charges = tmp_path / "charges.jsonl"
        model = build_model(str(snapshot)).eval()
        charges.write_text(
            "".join(
                json.dumps(r) + "\n" for r in predict_rows(model, tokenizer, smiles, 8)
            )
        )
        args.extend(["--charges", charges])
    output = tmp_path / "fig"
    # Packaged models load from the installed package without repository helpers.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    run_plot(
        "opt/design/plots.jl",
        [
            "--base",
            base,
            "--analysis",
            analysis,
            "--inputs",
            ROOT / "opt/design",
            "--output-dir",
            output,
            *args,
        ],
        "opt/design",
    )
    assert len(list(output.glob("*.pdf"))) == expected_figures
    assert len(list(output.glob("*.png"))) == expected_figures
    report = json.loads((output / "report.json").read_text())
    assert report["base"] == base
    assert report["models"] == {name: str(tmp_path / name) for name in heads}
    if analysis == "electrolytes":
        assert (output / "electrolytes_predictions.csv").is_file()


def test_sterochemistry_renders_combined_and_per_model_reports(tmp_path):
    records = [
        {
            "model": model,
            "label": label,
            "status": "sealed_test_complete",
            "mean_target_r2": score,
            "coverage_fraction": coverage,
        }
        for model, label, score, coverage in [
            ("MIST-28M", "mist-28m", 0.9, 1.0),
            ("MIST-1.8B", "mist-1p8b", 0.95, 1.0),
            ("SELFormer", "selformer", 0.8, 0.5),
        ]
    ]
    paths = []
    for record in records:
        path = tmp_path / (record["label"] + ".json")
        write_json(path, record)
        paths.append(path)
    output = tmp_path / "fig"
    run_plot(
        "opt/sterochemistry/plot.jl",
        ["--metrics", *paths, "--output-dir", output],
        "opt/sterochemistry",
    )
    for folder in [output, *(output / "models" / row["label"] for row in records)]:
        for name, extension in itertools.product(
            ("r2-comparison", "coverage"), ("pdf", "png")
        ):
            assert (folder / f"{name}.{extension}").stat().st_size > 100
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "complete"
    assert {row["model"] for row in report["models"]} == {
        row["model"] for row in records
    }
    assert (output / "results.tex").is_file()
