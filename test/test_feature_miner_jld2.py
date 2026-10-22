"""Opt-in regression tests for the existing Julia checkpoint/JLD2 path."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def make_probe_checkpoints(directory, *, legacy=False, layers=2):
    directory.mkdir(parents=True)
    locations = (
        ["output"]
        if legacy
        else ["output", "intermediate", "attention", "output.dense"]
    )
    names = [
        f"*encoder.layer.{layer}.{location}"
        for location in locations
        for layer in range(layers)
    ]
    state = {} if legacy else {"_prob_points": names}
    metrics = {}
    for index, name in enumerate(names):
        width = 16 if name.endswith("intermediate") else 8
        state[f"_probes.{index}.weight"] = (
            torch.arange(5 * width).reshape(5, width).float() + index * 100
        )
        state[f"_probes.{index}.bias"] = torch.arange(5).float() + index * 100
        metrics[f"val/{name}-loss"] = 0.12345678 + index / 100
        metrics[f"val/{name}-auroc"] = 0.98765432 - index / 100
    for name in names:
        prefix = name.replace(".", "-").replace("*", "star")
        torch.save(
            {
                "state_dict": state,
                "hyper_parameters": {
                    "probes": {"init_args": {"location": "output", "hidden_size": 8}}
                    if legacy
                    else None,
                    "model": {"init_args": {"name_or_path": "synthetic-encoder"}},
                },
                "datamodule_hyper_parameters": {
                    "init_args": {
                        "name_or_path": "lipinski",
                        "encoding": "smiles-kekule",
                        "tokenizer": "smirk",
                    }
                }
                if legacy
                else {
                    "path": "lipinski",
                    "encoding": "smiles-kekule",
                    "tokenizer": "smirk",
                },
                **({} if legacy else {"probe_metrics": metrics}),
                "epoch": 2,
                "global_step": 8000,
            },
            directory / f"{prefix}--epoch-2--loss-0.123--auroc-0.988.ckpt",
        )
    torch.save(
        {
            "training_steps": 9000,
            "requested_steps": 10_000,
            "step_budget_reached": False,
        },
        directory.parent / "fit.pt",
    )


@pytest.mark.skipif(
    os.environ.get("MIST_JULIA_SMOKE") != "1", reason="Opt-in Julia integration test"
)
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_loader_selects_exact_hooks_and_roundtrips_jld2(tmp_path, legacy):
    checkpoints = tmp_path / "pretrained/checkpoints"
    make_probe_checkpoints(checkpoints, legacy=legacy)
    result = subprocess.run(
        [
            "uv",
            "run",
            "julia",
            "--project=opt/FeatureMiner/plots",
            "--startup-file=no",
            "-e",
            r"""
            using FeatureMiner, DataFrames, JLD2, Test
            df, meta = FeatureMiner.load_fitted_probes(ARGS[1])
            legacy = ARGS[2] == "true"
            locations = legacy ? ["output"] : ["output", "intermediate", "attention", "output.dense"]
            @test nrow(df) == 2 * length(locations)
            @test meta.name_or_path == "synthetic-encoder"
            @test meta.dataset == "lipinski"
            for row in eachrow(df)
                index = 2 * (findfirst(==(row.location), locations) - 1) + row.layer - 1
                width = row.location == "intermediate" ? 16 : 8
                @test size(row.weight) == (5, width)
                @test row.weight[1, 1] == 4 * width + index * 100
                @test row.weight[2, 1] == index * 100
                @test row.bias == [4, 0, 1, 2, 3] .+ index * 100
                @test row.val_loss == (legacy ? 0.123 : 0.12345678 + index / 100)
                @test row.auroc == (legacy ? 0.988 : 0.98765432 - index / 100)
                @test row.selected_step == 8000
                @test row.training_steps == 9000
                @test row.requested_steps == 10000
                @test !row.step_budget_reached
            end
            output = joinpath(dirname(ARGS[1]), "probes.jld2")
            jldsave(output; probes=df)
            restored = load(output, "probes")
            @test isequal(restored, df)
            println("checkpoint/JLD2 roundtrip passed")
            """,
            str(checkpoints),
            str(legacy).lower(),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "UV_NO_SYNC": "1",
            "JULIA_PYTHONCALL_EXE": str(ROOT / "opt/FeatureMiner/.venv/bin/python"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
