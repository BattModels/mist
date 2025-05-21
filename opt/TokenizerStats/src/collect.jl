function tracked_stats()
    return (;
        fertility=CountMap(Int),
        nunique=CountMap(Int),
        out_of_vocab=Counter(Int),
        ngrams=ntuple(i -> CountMap(NTuple{i,Int}), 5),
    )
end

usage_stats(example, is_oov) = usage_stats!(tracked_stats(), example, is_oov)
function usage_stats!(stats, code::Vector{Int}, is_oov::Bool)

    # Track usage stats
    fit!(stats.fertility, length(code))
    fit!(stats.nunique, length(unique(code)))

    # Track n-grams
    for (n, s) in enumerate(stats.ngrams)
        fit!(s, SlidingWindow(code, n))
    end

    # Check for unknown tokens
    if is_oov
        fit!(stats.out_of_vocab, 1)
    end

    return stats
end

function leader_reduce(f, x; comm=MPI.COMM_WORLD)
    g = MPI.gather(x, comm; root=0)
    if MPI.Comm_rank(comm) == 0
        @assert length(g) == MPI.Comm_size(comm)
        return reduce(f, g)
    end
    return nothing
end

function rank_usage_stats(dataset::DatasetConfig, split::String; global_rank::Int, world_size::Int)
    ds = dataset_split(dataset, split; global_rank, world_size)
    _, tok_info = tokenizer(dataset)
    unk_token_id = tok_info.unk_token_id
    local_stats = tracked_stats()
    start_time = time()
    @info "rank $global_rank: started processing $split" now()
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        is_oov = unk_token_id in input_ids
        usage_stats!(local_stats, input_ids, is_oov)
        if idx % 1_000_000 == 0
            elapsed = time() - start_time
            @info "rank $global_rank on molecule $idx" idx elapsed idx / elapsed now()
        end
    end
    n_obs = nobs(local_stats[:fertility])
    elapsed = time() - start_time
    @info "Rank $global_rank has finished tokenizer stats" n_obs elapsed n_obs / elapsed now()

    return OnlineStats.Series(;
        fertility=local_stats.fertility,
        nunique=local_stats.nunique,
        out_of_vocab=local_stats.out_of_vocab,
        ngrams=OnlineStats.Series(local_stats.ngrams),
    )
end

function job_array_usage_stats(dataset::DatasetConfig, out_file::AbstractString; splits=["train"], world_size::Int=1, global_rank::Int=0)
    # Load Dataset and Tokenizer
    _, tok_info = tokenizer(dataset)
    out_file = out_file * "_split_$(join(splits, "_"))_rank_$global_rank.jld2"
    mkpath(dirname(out_file))
    jldopen(out_file * ".tmp", "w+") do f
        f["tokenizer"] = tok_info
        f["dataset"] = (; dataset_name=dataset_name(dataset), encoding=dataset.encoding)
    end

    for split in splits
        tokenizer_stats = rank_usage_stats(dataset, split; global_rank, world_size)
        @info "Saving results for $split on rank $global_rank" now()
        jldopen(out_file * ".tmp", "a+") do f
            serialize_usage!(f, split, tokenizer_stats)
        end
    end

    # Finalize
    mv(out_file * ".tmp", out_file; force=true)
    chmod(out_file, 0o444)

    return 0
end

function tabulate_dataset(dataset::DatasetConfig, out_file::AbstractString, splits::Vector{String}=["train"])
    # Setup mpi
    MPI.Init()
    comm = MPI.COMM_WORLD
    global_rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    @info "Rank $global_rank of $world_size is starting" now()

    # Load Dataset and Tokenizer
    _, tok_info = tokenizer(dataset)
    if global_rank == 0
        @debug "rank $global_rank: created $out_file" now()
        mkpath(dirname(out_file))
        jldopen(out_file * ".tmp", "w+") do f
            f["tokenizer"] = tok_info
            f["dataset"] = (; dataset_name=dataset_name(dataset), encoding=dataset.encoding)
        end
    end

    MPI.Barrier(comm)
    start_time = time()
    for split in splits
        rank_stats = rank_usage_stats(dataset, split; global_rank, world_size)
        tokenizer_stats = leader_reduce(merge!, rank_stats; comm)
        if global_rank == 0
            jldopen(out_file * ".tmp", "a+") do f
                serialize_usage!(f, split, tokenizer_stats)
            end
            @info "rank $global_rank: saved results for $split" now()
        end
    end

    if global_rank == 0
        jldopen(out_file * ".tmp", "a+") do f
            out_file["walltime"] = time() - start_time
            out_file["world_size"] = world_size
        end
        mv(out_file * ".tmp", out_file; force=true)
        chmod(out_file, 0o444)
        @info "Saved stats on rank $global_rank to $out_file" now()
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    @debug "rank $global_rank: finished" now()
    return 0
end

function model_loss(dataset::DatasetConfig, ref_file::String, output::String)
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    global_rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    @info "Rank $global_rank of $world_size is ready" now()

    # Create datamodule
    tok, tok_info = tokenizer(dataset)
    MPI.Barrier(comm)
    @info "Rank $global_rank of $world_size is starting" now()

    # Load Reference Tokenizer / n-gram model
    ngram, _, ref_info = load_ngram_model(ref_file)
    global_rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    # Init Fit Stats
    if global_rank == 0
        mkpath(dirname(output))
        jldopen(output * ".tmp", "w+") do f
            f["tokenizer"] = tok_info
            f["ref_tokenizer"] = ref_info
            f["dataset"] = (; dataset_name=dataset_name(dataset), encoding=dataset.encoding)
        end
    end

    MPI.Barrier(comm)
    start_time = time()
    for split in ["val", "train", "test"]
        ds = dataset_split(dataset, split; global_rank, world_size)
        stats = map(1:length(ngram)) do _
            OnlineStats.Series(;
                moments=OnlineStats.Moments(),
                extrema=Extrema(),
                histogram=KHist(100),
            )
        end |> OnlineStats.Group
        stats = (; cross_entropy=deepcopy(stats), cross_entropy_per_token=deepcopy(stats))

        @info "rank $global_rank: started processing $split" now()
        loss = zeros(length(ngram))
        start_time = time()
        for (idx, encoding) in enumerate(ds)
            code = pyconvert(Vector{Int}, encoding["input_ids"])
            for N in 1:length(ngram)
                loss[N] = cross_entropy(ngram, code; N)
            end
            fit!(stats.cross_entropy, tuple(loss))
            fit!(stats.cross_entropy_per_token, tuple(loss ./ length(code)))
            if idx % 1_000_000 == 0 && global_rank == 0
                elapsed = time() - start_time
                @info "rank $global_rank on molecule $idx" idx elapsed idx / elapsed
            end
        end
        # Reduce stats over ranks
        stats = leader_reduce(merge!, OnlineStats.Group(; stats...); comm)
        split_wall_time = time() - start_time
        if global_rank == 0
            jldopen(output * ".tmp", "a+") do f
                f[split] = (;
                    samples=nobs(stats),
                    cross_entropy=map(value, stats[:cross_entropy]),
                    cross_entropy_per_token=map(value, stats[:cross_entropy_per_token]),
                    walltime=split_wall_time,
                    world_size,
                )
            end
        end
        MPI.Barrier(comm)
    end

    if global_rank == 0
        jldopen(output * ".tmp", "a+") do f
            f["walltime"] = time() - start_time
        end
        mv(output * ".tmp", output; force=true)
        chmod(output, 0o444)
        @info "rank $global_rank: saved stats to $output" now()
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing

end

@annotate function avg_information_loss(dataset::DatasetConfig, ref_file::String, output::String; split="val")
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    global_rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    @info "Rank $global_rank of $world_size is starting" now()

    # Load Reference Tokenizer / n-gram model
    tok, tok_info = tokenizer(dataset)
    ngram, ref_tok, ref_info = load_ngram_model(ref_file)
    global_rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    if global_rank == 0
        mkpath(dirname(output))
        jldopen(output * ".tmp", "w+") do f
            f["tokenizer"] = tok_info
            f["ref_tokenizer"] = ref_info
            f["system"] = (; world_size)
        end
    end

    # Stats to track
    stats = map(1:length(ngram)) do _
        OnlineStats.Series(;
            moments=OnlineStats.Moments(),
            extrema=Extrema(),
            histogram=KHist(100),
        )
    end |> OnlineStats.Group
    info_loss = zeros(length(ngram))

    @info "rank $global_rank: started processing" now()
    ds = dataset_split(dataset, split; global_rank, world_size)

    MPI.Barrier(comm)
    start_time = time()
    for (idx, encoding) in enumerate(ds)
        info_loss = unk_information_loss(ngram, ref_tok, tok, encoding; smi_column="smi")
        fit!(stats, tuple(info_loss))
        if idx % 100 == 0 && global_rank == 0
            elapsed = time() - start_time
            @info "rank $global_rank on molecule $idx" idx elapsed idx / elapsed now()
        end
    end
    stats = leader_reduce(merge!, stats; comm)
    walltime = time() - start_time
    if global_rank == 0
        jldopen(output * ".tmp", "a+") do f
            f["samples"] = nobs(stats)
            f["info_loss"] = map(value, stats)
            f["walltime"] = walltime
        end
        mv(output * ".tmp", output; force=true)
        @info "saved results to $output" now()
        chmod(output, 0o444)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end
