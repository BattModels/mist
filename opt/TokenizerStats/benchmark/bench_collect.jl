using BenchmarkTools
using PythonCall
using TokenizerStats

const suite = BenchmarkGroup()

function fake_dataset(n=10)
    dir = mktempdir()
    mkpath(joinpath(dir, "data", "train"))
    mkpath(joinpath(dir, "data", "test"))
    mkpath(joinpath(dir, "data", "val"))
    src = joinpath(@__DIR__, "..", "..", "..", "smirk", "test", "opensmiles.smi")
    for split in ["train", "val", "test"]
        for i in 1:n
            cp(src, joinpath(dir, "data", split, "data_$i.txt"))
        end
    end
    return dir
end

const FAKE_DATASET_PATH = fake_dataset()

function setup_benchmark(tok_name; canonical=false)
    tokenizer = TokenizerStats.load_tokenizer(tok_name)
    ds = TokenizerStats.setup_dataset_mpi(FAKE_DATASET_PATH, tokenizer; canonical)
    ds = collect(Iterators.take(ds, 100))
    return tokenizer, ds
end

suite["tokenize"] = s = BenchmarkGroup()
batch = @py ["CC(NC)CC1=CC=C(OCO2)C2=C1"]
s["once"] = @benchmarkable TokenizerStats.batch_tokenize(batch, tokenizer; unk_token_id=0) setup=(tokenizer=TokenizerStats.load_tokenizer("smirk"))
s["raw_smiles"] = @benchmarkable foreach(identity, ds) setup=((tok, ds) = setup_benchmark("smirk"))

suite["stats"] = s = BenchmarkGroup()
example = (; input_ids = [1, 2, 3], text = "hey", is_oov = false)
stats = TokenizerStats.tracked_stats()
s["usage_stats"] = @benchmarkable TokenizerStats.usage_stats!($stats, $example)
