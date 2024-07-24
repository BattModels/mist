#!/bin/bash
set -e

export TOKENIZERS="
smirk
ibm/MoLFormer-XL-both-10pct
SmilesPE/SPE_ChEMBL
devalab/molgpt-moses
devalab/molgpt-guacamol
MolecularAI/Chemformer
MolecularAI/Chemformer-downstream
seyonec/ChemBERTa-zinc-base-v1
sagawa/ReactionT5-product-prediction
sagawa/ReactionT5-yield-prediction
rxn4chemistry/rxn_yields
rxn4chemistry/rxnfp
ChangwenXu98/TransPolymer
"

# Move to git root
cd $(git rev-parse --show-toplevel)

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source activate

# Precompile project
export JULIA_PKG_PRECOMPILE=0
srun -p venkvis-debug -c3 --mem 4G julia --color=yes --startup-file=no --project=opt/TokenizerStats -e '
    using Pkg
    Pkg.instantiate()
    using MPIPreferences
    @info "mpiexe" run(`which mpiexec`)
    MPIPreferences.use_system_binary()
    Pkg.precompile()
'
unset JULIA_PKG_PRECOMPILE

# Submit jobs for each tokenizer
for tok in $TOKENIZERS; do
    dataset="$(basename $1)"
    output="stats/$tok/stats-$dataset-canonical.json"
    config="{\"data\":\"$1\",\"tokenizer\":\"$tok\",\"output\":\"$output\"}"
    submit/submit.py opt/TokenizerStats/submit_tok_stats.j2 --no-default --no-confirm --json "$config" | sbatch
done
