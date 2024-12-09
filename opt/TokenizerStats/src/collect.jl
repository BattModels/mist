# Maximum number of oov samples to track
const MAX_OOV_SAMPLES = 100


function tracked_stats()
    return (;
        fertility=CountMap(Int),
        nunique=CountMap(Int),
        ngrams=ntuple(i -> CountMap(NTuple{i,Int}), 5),
        out_of_vocab=Counter(Int),
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

function setup_dm_mpi(dm::Py, split::AbstractString; rank::Int=0, size::Int=1)
    dm.trainer = (; global_rank=rank, world_size=size)
    dm.setup("fit")
    if split == "train"
        ds = dm.train_dataset
    elseif split == "val"
        ds = dm.val_dataset
    elseif split == "test"
        ds = dm.test_dataset
    else
        throw(ArgumentError(lazy"Invalid split: $split"))
    end
    return ds
end

function rank_usage_stats(datamodule, split; rank, size)
    ds = setup_dm_mpi(datamodule, split; rank, size)
    tokenizer = datamodule.tokenizer
    unk_token_id = pyconvert(Int, tokenizer.unk_token_id)
    local_stats = tracked_stats()
    start_time = time()
    @info "rank $rank: started processing $split" now()
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        is_oov = unk_token_id in input_ids
        usage_stats!(local_stats, input_ids, is_oov)
        if idx % 1_000_000 == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed now()
        end
    end
    n_obs = nobs(local_stats[:fertility])
    elapsed = time() - start_time
    @info "Rank $rank has finished tokenizer stats" n_obs elapsed n_obs / elapsed now()

    return OnlineStats.Series(;
        fertility=local_stats.fertility,
        nunique=local_stats.nunique,
        out_of_vocab=local_stats.out_of_vocab,
        ngrams=OnlineStats.Series(local_stats.ngrams),
    )
end

function job_array_usage_stats(datamodule::Py, out_file::AbstractString; tokenizer_name::AbstractString="", splits=["train"], size::Int=1, rank::Int=0)
    # Load Dataset and Tokenizer
    tokenizer = datamodule.tokenizer
    tokenizer_info = (;
        name=tokenizer_name,
        vocab_size=pyconvert(Int, length(tokenizer)),
        unk_token_id=pyconvert(Union{Int,Nothing}, tokenizer.unk_token_id),
    )

    splits = (length(splits) == 1 && first(splits) == "all") ? ["val", "train", "test"] : splits
    out_file = out_file * "_split_$(join(splits, "_"))_rank_$rank.jld2"
    mkpath(dirname(out_file))
    jldopen(out_file * ".tmp", "w+") do f
        f["tokenizer"] = tokenizer_info
    end

    for split in splits
        tokenizer_stats = rank_usage_stats(datamodule, split; rank, size)
        @info "Saving results for $split on rank $rank" now()
        jldopen(out_file * ".tmp", "a+") do f
            f[split] = (;
                samples=nobs(tokenizer_stats),
                map(value, tokenizer_stats.stats)...
            )
        end
    end

    # Finalize
    mv(out_file * ".tmp", out_file; force=true)
    chmod(out_file, 0o444)

    return 0
end

function tabulate_dataset(datamodule::Py, out_file::AbstractString; tokenizer_name::AbstractString="", splits=["train"])
    # Setup mpi
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting" now()

    # Load Dataset and Tokenizer
    tokenizer = datamodule.tokenizer
    tokenizer_info = (;
        name=tokenizer_name,
        vocab_size=pyconvert(Int, length(tokenizer)),
        unk_token_id=pyconvert(Union{Int,Nothing}, tokenizer.unk_token_id),
    )
    splits = (length(splits) == 1 && first(splits) == "all") ? ["val", "train", "test"] : splits
    if rank == 0
        @debug "rank $rank: created $out_file" now()
        mkpath(dirname(out_file))
        jldopen(out_file * ".tmp", "w+") do f
            f["tokenizer"] = tokenizer_info
        end
    end

    MPI.Barrier(comm)
    for split in splits
        rank_stats = rank_usage_stats(datamodule, split; rank, size)
        tokenizer_stats = leader_reduce(merge!, rank_stats; comm)
        if rank == 0
            jldopen(out_file * ".tmp", "a+") do f
                f[split] = (;
                    samples=nobs(tokenizer_stats),
                    map(value, tokenizer_stats.stats)...
                )
            end
            @info "rank $rank: saved results for $split" now()
        end
    end

    if rank == 0
        mv(out_file * ".tmp", out_file; force=true)
        chmod(out_file, 0o444)
        @info "Saved stats on rank $rank to $out_file" now()
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    @debug "rank $rank: finished" now()
    return 0
end

function model_loss(datamodule::Py, ref_file::String, output::String)
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting" now()

    # Load Reference Tokenizer / n-gram model
    ngram, ref_tok, ref_info = load_ngram_model(ref_file)
    rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    # Init Fit Stats
    tok = datamodule.tokenizer
    if rank == 0
        mkpath(dirname(output))
        jldopen(output * ".tmp", "w+") do f
            f["tokenizer"] = (;
                vocab_size=pyconvert(Int, length(tok)),
                unk_token_id=pyconvert(Union{Int,Nothing}, tok.unk_token_id),
            )
            f["ref_tokenizer"] = ref_info
        end
    end

    MPI.Barrier(comm)
    for split in ["val", "train", "test"]
        ds = setup_dm_mpi(datamodule, split; rank, size)
        stats = map(1:length(ngram)) do _
            OnlineStats.Series(;
                moments=OnlineStats.Moments(),
                extrema=Extrema(),
                histogram=KHist(100),
            )
        end |> OnlineStats.Group
        stats = (; kld=deepcopy(stats), kld_per_token=deepcopy(stats))

        @info "rank $rank: started processing $split" now()
        loss = zeros(length(ngram))
        start_time = time()
        for (idx, encoding) in enumerate(ds)
            code = pyconvert(Vector{Int}, encoding["input_ids"])
            for N in 1:length(ngram)
                loss[N] = autoregressive_kld(ngram, code; N)
            end
            fit!(stats.kld, tuple(loss))
            fit!(stats.kld_per_token, tuple(loss ./ length(code)))
            if idx % 1_000_000 == 0 && rank == 0
                elapsed = time() - start_time
                @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
            end
        end
        # Reduce stats over ranks
        stats = leader_reduce(merge!, OnlineStats.Group(; stats...); comm)
        if rank == 0
            jldopen(output * ".tmp", "a+") do f
                f[split] = (;
                    samples=nobs(stats),
                    kld=map(value, stats[:kld]),
                    kld_per_token=map(value, stats[:kld_per_token]),
                )
            end
        end
        MPI.Barrier(comm)
    end

    if rank == 0
        mv(output * ".tmp", output; force=true)
        chmod(output, 0o444)
        @info "rank $rank: saved stats to $output" now()
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing

end

@annotate function avg_information_loss(datamodule::Py, ref_file::String, output::String)
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting" now()

    # Load Reference Tokenizer / n-gram model
    tokenizer = datamodule.tokenizer
    ngram, ref_tok, ref_info = load_ngram_model(ref_file)
    rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    if rank == 0
        mkpath(dirname(output))
        jldopen(output * ".tmp", "w+") do f
            tok = datamodule.tokenizer
            f["tokenizer"] = (;
                vocab_size=pyconvert(Int, length(tok)),
                unk_token_id=pyconvert(Union{Int,Nothing}, tok.unk_token_id),
            )
            f["ref_tokenizer"] = ref_info
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

    @info "rank $rank: started processing" now()
    ds = setup_dm_mpi(datamodule, "val"; rank, size)

    MPI.Barrier(comm)
    smi_column = pyconvert(String, datamodule.smi_column)
    start_time = time()
    for (idx, encoding) in enumerate(ds)
        info_loss = unk_information_loss(ngram, ref_tok, tokenizer, encoding; smi_column)
        fit!(stats, tuple(info_loss))
        if idx % 100 == 0 && rank == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed now()
        end
    end
    stats = leader_reduce(merge!, stats; comm)
    if rank == 0
        jldopen(output * ".tmp", "a+") do f
            f["samples"] = nobs(stats)
            f["info_loss"] = map(value, stats)
        end
        mv(output * ".tmp", output; force=true)
        @info "saved results to $output" now()
        chmod(output, 0o444)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end

