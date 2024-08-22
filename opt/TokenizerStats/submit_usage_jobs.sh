#!/bin/bash
set -e

export TOKENIZERS="
smirk
character
ibm/MoLFormer-XL-both-10pct-oov
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
../../smirk-gpe-50k-mb-ss
../../smirk-gpe-50k-nmb-ss
../../smirk-gpe-small-50k-mb-ss
google/gemma-7b
Xenova/gpt-4o
meta-llama/Meta-Llama-3.1-8B
meta-llama/Meta-Llama-3-8B
"

# Move to TokenizerStats dir
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source ./activate

# Precompile project
if [[ ! -f Manifest.toml ]]; then
    export JULIA_PKG_PRECOMPILE=0
    srun -p venkvis-debug -c2 --mem 4G julia --color=yes --startup-file=no --project -e '
        using Pkg
        Pkg.resolve()
        Pkg.instantiate()
        using MPIPreferences
        @info "mpiexe" run(`which mpiexec`)
        MPIPreferences.use_system_binary()
        Pkg.precompile()
    '
    unset JULIA_PKG_PRECOMPILE
fi

# Submit jobs for each tokenizer + dataset
set -x
for tok in $TOKENIZERS; do
    sbatch --time 2-0:0:0 -n 64 ./submit_tok_stats.sh usage /nfs/turbo/coe-venkvis/mist/realspace_v4_dev $tok
    for dataset in qm8 qm9 esol freesolv lipo muv hiv bace bbbp tox21 toxcast sider clintox; do
        sbatch -n 1 ./submit_tok_stats.sh usage $dataset $tok
    done
done
