# BayesianScaling

A Julia Package for fitting regression models using MCMC, that was used to fit penalized neural scaling laws.
To install:

- Install Julia: https://julialang.org/downloads/
- Instantiate the package: `julia --project -e 'using Pkg; Pkg.instantiate()`
- Download the wandb records or chains ([doi:10.5281/zenodo.17527149](https://doi.org/10.5281/zenodo.17527149))

## Code Organization

- `./scripts/` are used for fitting and analyzing the neural scaling laws.
- `./plots/` has plotting code for the paper and various conferences
- `./src` is the MCMC regression and analysis package powering this work
    - [ppl.jl](./src/ppl.jl): Define a regression first interface for fitting MCMC models,
        plus single-pass algorithms for working with posterior samples
    - [scaling.jl](./src/scaling.jl): functional forms for neural scaling laws and derived qualities
    - [analysis.jl](./src/analysis.jl.jl): Code for predicting the perform of models using fitted neural scaling laws
- `./test/` has the unit tests for the BayesianScaling.jl package
- `./benchmark/`: benchmark suite for evaluating different AD backends using [PkgJogger.jl](https://github.com/awadell1/PkgJogger.jl)

## Paper figure workflow

`workflow.pomset` stages the checksum-pinned Zenodo `21302713/scaling.tar.gz`
archive and runs the original Julia renderers over its 62 posterior chains.
The figure task declares only `scaling_panel_baseline.pdf`, which is consumed
by the main scaling composition. The renderer still writes all alternative-fit,
parity, and PNG plots under `results/scaling/figures`. The separate summary task
declares `sweep_smoothing` and the parameter and exponent tables; the campaign
table is a separate TeX artifact. `FIGURES`, `TABLES`, and `CHAINS` are exported
for downstream use. No model weights are
downloaded and no posterior fitting or foundation-model training is performed.

The native commands now accept explicit output paths; their old defaults remain:

```bash
julia --project=plots plots/paper.jl CHAIN_ROOT OUTPUT_DIR
julia --project=plots plots/param_table.jl CHAIN_ROOT OUTPUT_DIR
julia --project=plots plots/campaign.jl CHAIN_FILE OUTPUT_TEX
```

Replaying these released posteriors does not require the unavailable original
training histories. Independently refitting them or reproducing all training-cost
claims does require additional evidence:

- `data/user/wandb-artifacts/`: original run configurations and complete metric
  histories for the 138 selected campaign runs, including the validation windows
  used for the smoothed-loss fits.
- `data/user/scaling-runtime.csv`: `run_id`, runtime, and GPU/world-size accounting
  for those runs, required to recompute the reported GPU-hours.
- `data/user/gsd-provenance.json`: candidate-generation code revision, RNG seed,
  candidate ordering, selected indices, and full-factorial constraints/cost
  formula. The currently available full-factorial calculation does not recover
  the paper's reported 27,100 PF-days.

These are provenance requirements for an independent campaign reconstruction,
not substitute figure inputs. The renderer retains its original posterior
subsampling behavior, so released-chain replay does not promise identical random
draw selection or byte-identical PDFs.
