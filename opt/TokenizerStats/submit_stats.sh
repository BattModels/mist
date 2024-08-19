#!/bin/bash
set -e

export TOKENIZERS="
smirk
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
"

# Move to TokenizerStats dir
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source ./activate

# Precompile project
export JULIA_PKG_PRECOMPILE=0
srun -p venkvis-debug -c3 --mem 4G julia --color=yes --startup-file=no --project -e '
    using Pkg
    Pkg.instantiate()
    using MPIPreferences
    @info "mpiexe" run(`which mpiexec`)
    MPIPreferences.use_system_binary()
    Pkg.precompile()
'
unset JULIA_PKG_PRECOMPILE

# Submit jobs for each tokenizer + dataset
for tok in $TOKENIZERS; do
    sbatch ./submit_tok_stats.sh /nfs/turbo/coe-venkvis/mist/realspace_v4_dev $tok
    for dataset in qm8 qm9 esol freesolv lipo muv hiv bace bbbp tox21 toxcast sider clintox; do
        sbatch ./submit_tok_stats.sh $dataset $tok
    done
done
