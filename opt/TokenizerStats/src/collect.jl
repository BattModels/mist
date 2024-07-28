# Maximum number of oov samples to track
const MAX_OOV_SAMPLES = 100


function tracked_stats()
    return (;
        fertility=CountMap(Int),
        nunique=CountMap(Int),
        token_usage=CountMap(Int),
        out_of_vocab=Counter(Int),
        oov_samples=Set{String}(),
        distict_samples=HyperLogLog(String)
    )
end

usage_stats(example, unk_token_id::Int) = usage_stats!(tracked_stats(), example, unk_token_id)
function usage_stats!(stats, example, unk_token_id::Int)
    code = pyconvert(Vector{Int64}, example["input_ids"])

    # Track usage stats
    fit!(stats.fertility, length(code))
    fit!(stats.nunique, length(unique(code)))
    fit!(stats.token_usage, code)
    fit!(stats.distict_samples, pyconvert(String, example["text"].strip()))

    # Check for unknown tokens
    if unk_token_id in example["input_ids"] || pyconvert(Bool, example["text"].strip() != example["decode"].strip())
        fit!(stats.out_of_vocab, 1)
        if length(stats.oov_samples) < MAX_OOV_SAMPLES
            push!(stats.oov_samples, pyconvert(String, example["text"]))
        end
    end

    return stats
end

function tokenize(batch; tokenizer)
    out = tokenizer(batch["text"])
    out["decode"] = tokenizer.batch_decode(out["input_ids"], skip_special_tokens=true)
    out["text"] = batch["text"]
    return out
end

shannon_entropy(p::Real) = -p * log.(p)

function shannon_entropy!(stats, example; token_entropy::Dict{Int, V}) where {V <: Real}
    code = pyconvert(Vector{Int64}, example["input_ids"])
    H = sum(Base.Fix1(getindex, token_entropy), code; init=zero(V))
    fit!(stats, (H, H / length(code)))
    return stats
end

function leader_reduce(f, x)
    comm = MPI.COMM_WORLD
    g = MPI.gather(x, comm; root=0)
    if MPI.Comm_rank(comm) == 0
        return reduce(f, g)
    end
    return nothing
end

function setup_dataset_mpi(ds_path, tokenizer; rank=0, size=1, canonical=false)
    ds_path = realpath(expanduser(ds_path))
    ds = load_dataset(ds_path, split="train", streaming=true, keep_in_memory=false)
    if canonical
        ds = ds.map(x -> pydict(; text=rdkit_canonical(x["text"])), batched=false)
        ds = ds.filter(x -> pybool(x["text"]))
    end
    ds = ds.map(tokenize,
        batched=true,
        batch_size=1000,
        fn_kwargs=Dict("tokenizer" => tokenizer)
    )
    return split_dataset_by_node(ds, rank, size)
end

function tabulate_dataset(ds_path, tok_name, out_file; canonical=false)
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
        vocab_size=pyconvert(Int, tokenizer.vocab_size),
        unk_token_id=pyconvert(Int, tokenizer.unk_token_id),
    )
    local_stats = tracked_stats()
    ds = setup_dataset_mpi(ds_path, tokenizer; rank, size, canonical)
    for example in ds
        usage_stats!(local_stats, example, tokenizer_info.unk_token_id)
    end
    @info "Rank $rank has finished tokenizer stats"
    local_tok_stats = Series(; Base.structdiff(local_stats, NamedTuple{(:oov_samples,)})...)
    tokenizer_stats = leader_reduce(merge!, local_tok_stats)
    oov_samples = leader_reduce(union, local_stats.oov_samples)

    # Broadcast token usage to all ranks
    token_usage = rank == 0 ? value(tokenizer_stats[:token_usage]) : nothing
    token_usage = MPI.bcast(token_usage, comm; root=0)
    n_tokens = sum(values(token_usage))

    # Tabulate the entropy of each token
    token_entropy = Dict(k => shannon_entropy(v / n_tokens) for (k, v) in token_usage)

    # Compute entropy statistics for the dataset
    @info "Rank $rank: Computing tokenizer entropy"
    entropy = Series(; moments=OnlineStats.Moments(), extrema=Extrema(), hist=KHist(100))
    entropy = Group(; per_molecule=deepcopy(entropy), per_token=deepcopy(entropy))
    for example in ds
        shannon_entropy!(entropy, example; token_entropy)
    end
    entropy = leader_reduce(merge!, entropy)
    @info "Rank $rank: Finished tokenizer entropy"

    if rank == 0
        @info "Tabulating stats on rank $rank"
        stats = (;
            tokenizer=tokenizer_info,
            samples=nobs(tokenizer_stats[:fertility]),
            oov_samples,
            entropy=map(value, entropy.stats),
            map(value, tokenizer_stats.stats)...
        )
        mkpath(dirname(out_file))
        open(out_file, "w") do io
            JSON.print(io, stats)
        end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return 0
end
