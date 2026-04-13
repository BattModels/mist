# Synthetic Accessibility

Evaluating if MIST's molecular surprise correlates with a chemist's ``Synthetic Accessibility''.
Evaluates multiple metrics for synthetic accessibility on two datasets reporting results, plus analysis/plotting code to access the results.

# Installation

1. Install [Julia](https://julialang.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Evaluate the metrics: `uv run python main.py`
3. Generate plots & analysis results: `julia --project ./plot.jl`
