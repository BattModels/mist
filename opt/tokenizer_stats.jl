#!/usr/bin/env -S julia --project
using PythonCall
using OnlineStats: Counter, CountMap, Series, fit!, value, nobs
using MPI: MPI
using JSON: JSON

# Load Python Dependencies
@time begin
    const load_tokenizer = pyimport("electrolyte_fm.utils.tokenizer").load_tokenizer
    const load_dataset = pyimport("datasets").load_dataset
    const split_dataset_by_node = pyimport("datasets.distributed").split_dataset_by_node
end

function tracked_stats()
    return Series(;
        fertility = CountMap(Int),
        nunique = CountMap(Int),
        token_usage = CountMap(Int),
    )
end

usage_stats(example) = usage_stats!(tracked_stats(), example)
function usage_stats!(stats, example)
    code = pyconvert(Vector{Int64}, example["input_ids"])
    input = pyconvert(String, example["text"])
    fit!(stats[:fertility], length(code))
    fit!(stats[:nunique], length(unique(code)))
    fit!(stats[:token_usage], code)
    return stats
end

tokenize(batch; tokenizer) = tokenizer(batch["text"])

function entropy!(stats, example; token_usage::)


end

function tabulate_dataset(ds_path, tok_name)
    # Setup mpi
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting"

    # Load Dataset and Tokenizer
    tokenizer = load_tokenizer(tok_name)
    ds_path = realpath(expanduser(ds_path))
    ds = load_dataset(ds_path, split="train", streaming=true, keep_in_memory=false)
    ds = ds.map(tokenize,
        batched=true,
        batch_size=1000,
        fn_kwargs=Dict("tokenizer" => tokenizer)
    )


    ds = split_dataset_by_node(ds, rank, size)
    local_stats = tracked_stats()
    for example in Iterators.take(ds, 10000)
        usage_stats!(local_stats, example)
    end
    @info "Rank $rank has finished" local_stats

    # Collect Statistics
    rank_stats = MPI.gather(local_stats, comm; rank = 0)
    if rank == 0
        @info "Rank $rank: Gathered results"
        collective_stats = reduce(merge!, rank_stats)
        @info stats
        open("stats.json", "w") do fid
            stats = map(value, collective_stats.stats)
            stats = (; stats... samples=nobs(collective_stats[:fertility]))
            JSON.print(fid, map(value, stats.stats))
        end
    end
    MPI.Finalize()
    return nothing
end

function main(args::Vector{String})
    @assert length(args) == 2
    ds_path = args[1]
    tok_name = args[2]
    tabulate_dataset(ds_path, tok_name)
end

!isinteractive() && exit(main(ARGS))
