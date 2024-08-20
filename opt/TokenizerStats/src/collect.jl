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

usage_stats(example, is_oov) = usage_stats!(tracked_stats(), example, is_oov)
function usage_stats!(stats, code::Vector{Int}, is_oov::Bool)

    # Track usage stats
    fit!(stats.fertility, length(code))
    fit!(stats.nunique, length(unique(code)))
    fit!(stats.token_usage, code)

    # Check for unknown tokens
    if is_oov
        fit!(stats.out_of_vocab, 1)
    end

    return stats
end

function batch_tokenize(text::Py, tokenizer::Py; unk_token_id::Integer)
    code = tokenizer(text)["input_ids"]

    # Convert to julia
    code_jl = pyconvert(Vector{Vector{Int}}, code)
    text = pyconvert(Vector{String}, text)
    is_oov = unk_token_id in code_jl
    return Iterators.map(zip(code_jl, text, is_oov)) do (code, text, is_oov)
        (; code, text, is_oov)
    end
end

function leader_reduce(f, x)
    comm = MPI.COMM_WORLD
    g = MPI.gather(x, comm; root=0)
    if MPI.Comm_rank(comm) == 0
        return reduce(f, g)
    end
    return nothing
end

function setup_dm_mpi(dm::Py, split::AbstractString; rank::Int=0, size::Int=1)
    if split == "train"
        ds = dm.train_dataset
    elseif split == "val"
        ds = dm.val_dataset
    elseif split == "test"
        ds = dm.test_dataset
    else
        throw(ArgumentError(lazy"Invalid split: $split"))
    end
    return split_dataset_by_node(ds, rank, size)
end

function tabulate_dataset(datamodule::Py, out_file::AbstractString; tokenizer_name::AbstractString="")
    # Setup mpi
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting"

    # Load Dataset and Tokenizer
    tokenizer = datamodule.tokenizer
    tokenizer_info = (;
        name=tokenizer_name,
        vocab_size=pyconvert(Int, tokenizer.vocab_size),
        unk_token_id=pyconvert(Int, tokenizer.unk_token_id),
    )
    local_stats = tracked_stats()
    ds = setup_dm_mpi(datamodule, "train"; rank, size)
    collator = datamodule.data_collator
    start_time = time()
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        is_oov = tokenizer_info.unk_token_id in input_ids
        usage_stats!(local_stats, input_ids, is_oov)
        if idx % 1_000_000 == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
        end
    end
    n_obs = nobs(local_stats[:fertility])
    elapsed = time() - start_time
    @info "Rank $rank has finished tokenizer stats" n_obs elapsed n_obs / elapsed

    # Reduce stats over ranks
    local_tok_stats = OnlineStats.Series(; Base.structdiff(local_stats, NamedTuple{(:oov_samples,)})...)
    tokenizer_stats = leader_reduce(merge!, local_tok_stats)
    oov_samples = leader_reduce(union, local_stats.oov_samples)

    # Broadcast token usage to all ranks
    token_usage = rank == 0 ? value(tokenizer_stats[:token_usage]) : nothing
    token_usage = MPI.bcast(token_usage, comm; root=0)
    n_tokens = sum(values(token_usage))

    if rank == 0
        @info "Tabulating stats on rank $rank"
        stats = (;
            tokenizer=tokenizer_info,
            samples=nobs(tokenizer_stats[:fertility]),
            oov_samples,
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
