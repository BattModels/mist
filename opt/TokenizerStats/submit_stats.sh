#!/bin/bash

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

for tok in $TOKENIZERS; do
    dataset="$(basename $1)"
    output="stats/$tok/stats-$dataset.json"
    config="{\"data\":\"$1\",\"tokenizer\":\"$tok\",\"output\":\"$output\"}"
    submit/submit.py submit/submit_tok_stats.j2 --no-confirm --json "$config" | sbatch
done
