# Maximum number of oov samples to track
const MAX_OOV_SAMPLES = 100


function tracked_stats()
    return (;
        fertility=CountMap(Int),
        nunique=CountMap(Int),
        ngrams=ntuple(i -> CountMap(NTuple{i, Int}), 5),
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

function rank_usage_stats(datamodule, split; rank, size)
    ds = setup_dm_mpi(datamodule, split; rank, size)
    tokenizer = datamodule.tokenizer
    unk_token_id = pyconvert(Int, tokenizer.unk_token_id)
    local_stats = tracked_stats()
    start_time = time()
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        is_oov = unk_token_id in input_ids
        usage_stats!(local_stats, input_ids, is_oov)
        if idx % 1_000_000 == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
        end
    end
    n_obs = nobs(local_stats[:fertility])
    elapsed = time() - start_time
    @info "Rank $rank has finished tokenizer stats" n_obs elapsed n_obs / elapsed

    return OnlineStats.Series(;
        fertility=local_stats.fertility,
        nunique=local_stats.nunique,
        out_of_vocab=local_stats.out_of_vocab,
        ngrams=OnlineStats.Series(local_stats.ngrams),
    )
end

function tabulate_dataset(datamodule::Py, out_file::AbstractString; tokenizer_name::AbstractString="", splits=["train"])
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
        vocab_size=pyconvert(Int, length(tokenizer)),
        unk_token_id=pyconvert(Union{Int, Nothing}, tokenizer.unk_token_id),
    )
    stats = Dict{Symbol, Any}()
    splits = (length(splits) == 1 && first(splits) == "all") ? ["train", "val", "test"] : splits
    for split in splits
        @show rank_stats = rank_usage_stats(datamodule, split; rank, size)
        # tokenizer_stats = leader_reduce(merge!, local_stats)
        tokenizer_stats = rank_stats
        if rank == 0 || true
            @info "Saving results for $split on rank $rank"
            stats[Symbol(split)] = (;
                samples=nobs(tokenizer_stats),
                map(value, tokenizer_stats.stats)...
            )
        end
    end

    # Save stats
    if rank == 0 || true
        out_file = out_file * "_split_$(join(splits, "_"))_rank_$rank.bson"
        @info "Saving stats on rank $rank to $out_file"
        mkpath(dirname(out_file))
        stats[:tokenizer] = tokenizer_info
        rm(out_file; force=true)
        BSON.bson(out_file; stats...)
        chmod(out_file, 0o444)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return 0
end

function model_loss(datamodule::Py, ref_file::String, output::String)
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting"

    # Load Reference Tokenizer / n-gram model
    ngram, ref_tok, ref_info = load_ngram_model(ref_file)
    rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    # Init Fit Stats
    tok = datamodule.tokenizer
    fit_stats = Dict{Symbol, Any}(
        :tokenizer => (;
            vocab_size=pyconvert(Int, length(tok)),
            unk_token_id=pyconvert(Union{Int, Nothing}, tok.unk_token_id),
        ),
        :ref_tokenizer => ref_info,
    )

    for split in ["train", "val", "test"]
        ds = setup_dm_mpi(datamodule, split; rank, size)
        unk_token_id = pyconvert(Int, datamodule.tokenizer.unk_token_id)
        stats = map(1:length(ngram)) do _
            OnlineStats.Series(;
                moments=OnlineStats.Moments(),
                extrema=Extrema(),
                histogram=KHist(100),
            )
        end |> OnlineStats.Group
        stats = (; kld=deepcopy(stats),)

        @info "rank $rank: started processing $split"
        loss = zeros(length(ngram))
        start_time = time()
        for (idx, encoding) in enumerate(ds)
            code = pyconvert(Vector{Int}, encoding["input_ids"])
            for N in 1:length(ngram)
                loss[N] = autoregressive_kld(ngram, code; N)
            end
            fit!(stats.kld, tuple(loss))
            if idx % 1_000_000 == 0 && rank == 0
                elapsed = time() - start_time
                @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
            end
        end
        # Reduce stats over ranks
        stats = leader_reduce(merge!, OnlineStats.Group(; stats...))
        if rank == 0
            fit_stats[Symbol(split)] = (;
                samples=nobs(stats),
                kld=map(value, stats[:kld]),
            )
        end
        MPI.Barrier(comm)
    end

    if rank == 0
        @info "rank $rank: saving stats to $output" fit_stats
        mkpath(dirname(output))
        rm(output; force=true)
        BSON.bson(output; fit_stats...)
        chmod(output, 0o444)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing

end

function avg_information_loss(datamodule::Py, ref_file::String, output::String)
    # Init MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    @info "Rank $rank of $size is starting"

    # Load Reference Tokenizer / n-gram model
    tokenizer = datamodule.tokenizer
    ngram, ref_tok, ref_info = load_ngram_model(ref_file)
    rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file ($(ref_info.sha256[1:8]))"

    # Stats to track
    stats = map(1:length(ngram)) do _
        OnlineStats.Series(;
            moments=OnlineStats.Moments(),
            extrema=Extrema(),
            histogram=KHist(100),
        )
    end |> OnlineStats.Group
    info_loss = zeros(length(ngram))

    @info "rank $rank: started processing"
    ds = setup_dm_mpi(datamodule, "val"; rank, size)
    start_time = time()
    for (idx, encoding) in enumerate(ds)
        info_loss = unk_information_loss(ngram, ref_tok, tokenizer, encoding)
        fit!(stats, tuple(info_loss))
        if idx % 10 == 0 && rank == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
        end
    end
    stats = leader_reduce(merge!, stats)
    if rank == 0
        @info "saving results to $output"
        tok = datamodule.tokenizer
        stats = (;
            tokenizer=(;
                vocab_size=pyconvert(Int, length(tok)),
                unk_token_id=pyconvert(Union{Int, Nothing}, tok.unk_token_id),
            ),
            ref_tokenizer=ref_info,
            samples=nobs(stats),
            info_loss=map(value, stats),
        )
        mkpath(dirname(output))
        rm(output; force=true)
        BSON.bson(output; stats...)
        chmod(output, 0o444)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end
