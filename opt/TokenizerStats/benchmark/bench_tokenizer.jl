using BenchmarkTools
using TokenizerStats: DatasetConfig, dataset_split
using PythonCall: pyconvert

const suite = BenchmarkGroup()

TOKENIZERS = [
    "smirk",
    "character",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "SmilesPE/SPE_ChEMBL",
    "devalab/molgpt-moses",
    "devalab/molgpt-guacamol",
    "MolecularAI/Chemformer",
    "MolecularAI/Chemformer-downstream",
    "seyonec/ChemBERTa-zinc-base-v1",
    "sagawa/ReactionT5-product-prediction",
    "sagawa/ReactionT5-yield-prediction",
    "rxn4chemistry/rxn_yields",
    "rxn4chemistry/rxnfp",
    "ChangwenXu98/TransPolymer",
    "../../smirk-gpe-50k-mb-ss",
    "../../smirk-gpe-50k-nmb-ss",
    "../../smirk-gpe-small-50k-mb-ss",
    "google/gemma-7b",
    "Xenova/gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3-8B",
]

function setup_dataset(tokenizer, encoding)
    dc = DatasetConfig("qm9", tokenizer, encoding)
    ds = dataset_split(dc, "val")
    return Iterators.take(ds, 1000)
end
for name in TOKENIZERS
    suite[name] = tok_suite = BenchmarkGroup()
    for encoding in ["smiles", "smiles-canonical", "smiles-kekule"]
        tok_suite[encoding] = @benchmarkable foreach(identity, ds) setup=(ds = setup_dataset($name, $encoding))
    end
end
