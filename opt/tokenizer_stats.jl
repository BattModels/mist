#!/usr/bin/env -S julia --project
using PythonCall
using OnlineStats: Counter, CountMap, Series, Moments, Extrema, P2Quantile, fit!, value, nobs
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
        fertility=CountMap(Int),
        nunique=CountMap(Int),
        token_usage=CountMap(Int),
        out_of_vocab=Counter(Int),
    )
end

usage_stats(example) = usage_stats!(tracked_stats(), example)
function usage_stats!(stats, example, unk_token_id::Int)
    code = pyconvert(Vector{Int64}, example["input_ids"])

    # Track usage stats
    fit!(stats[:fertility], length(code))
    fit!(stats[:nunique], length(unique(code)))
    fit!(stats[:token_usage], code)

    # Check for unknown tokens
    unk_token_id in code && fit!(stats[:out_of_vocab], 1)

    return stats
end

tokenize(batch; tokenizer) = tokenizer(batch["text"])

shannon_entropy(p::Real) = -p * log.(p)

function shannon_entropy!(stats, example; token_entropy::Dict{Int,<:Real})
    code = pyconvert(Vector{Int64}, example["input_ids"])
    H = sum(Base.Fix1(getindex, token_entropy), code; init=zero(valtype(token_entropy)))
    fit!(stats, H)
    return stats
end

""" Merge stats from all ranks on rank 0 """
function reduce_stats(stat)
    comm = MPI.COMM_WORLD
    g = MPI.gather(stat, comm; root=0)
    if MPI.Comm_rank(comm) == 0
        return reduce(merge!, g)
    else
        return nothing
    end
end

function tabulate_dataset(ds_path, tok_name, out_file)
    # Setup mpi
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting"

    # Load Dataset and Tokenizer
    tokenizer = load_tokenizer(tok_name)
    tokenizer_info = (;
        name=tok_name,
        vocab_size=length(tokenizer.vocab),
        unk_token_id=pyconvert(Int, tokenizer.unk_token_id),
    )
    vocab_size=length(tokenizer.vocab)
    ds_path = realpath(expanduser(ds_path))
    ds = load_dataset(ds_path, split="train", streaming=true, keep_in_memory=false)
    ds = ds.map(tokenize,
        batched=true,
        batch_size=1000,
        fn_kwargs=Dict("tokenizer" => tokenizer)
    )

    # Compute Usage Statistics
    ds = split_dataset_by_node(ds, rank, size)
    local_stats = tracked_stats()
    for example in ds
        usage_stats!(local_stats, example, tokenizer_info.unk_token_id)
    end
    @info "Rank $rank has finished tokenizer stats"
    tokenizer_stats = reduce_stats(local_stats)

    # Broadcast token usage to all ranks
    token_usage = rank == 0 ? value(tokenizer_stats[:token_usage]) : nothing
    token_usage = MPI.bcast(token_usage, comm; root=0)
    n_tokens = sum(values(token_usage))

    # Tabulate the entropy of each token
    token_entropy = Dict(k => shannon_entropy(v / n_tokens) for (k, v) in token_usage)

    # Compute entropy statistics for the dataset
    @info "Rank $rank: Computing tokenizer entropy"
    entropy = Series(; moments=Moments(), extrema=Extrema())
    for example in ds
        shannon_entropy!(entropy, example; token_entropy)
    end
    entropy = reduce_stats(entropy)
    @info "Rank $rank: Finished tokenizer entropy"

    if rank == 0
        @info "Tabulating stats on rank $rank"
        stats = (;
            tokenizer=tokenizer_info,
            samples=nobs(tokenizer_stats[:fertility]),
            entropy=map(value, entropy.stats),
            map(value, tokenizer_stats.stats)...
        )
        open(out_file, "w") do io
            JSON.print(io, stats)
        end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return 0
end

function main(args::Vector{String})
    @assert length(args) >= 2
    ds_path = args[1]
    tok_name = args[2]
    out_file = length(args) == 3 ? args[3] : "stats.json"
    tabulate_dataset(ds_path, tok_name, out_file)
end

!isinteractive() && exit(main(ARGS))
