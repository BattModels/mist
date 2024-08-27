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
        vocab_size=pyconvert(Int, length(tokenizer)),
        unk_token_id=pyconvert(Union{Int, Nothing}, tokenizer.unk_token_id),
    )
    local_stats = tracked_stats()
    ds = setup_dm_mpi(datamodule, "train"; rank, size)
    start_time = time()
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        is_oov = tokenizer_info.unk_token_id in input_ids
        usage_stats!(local_stats, input_ids, is_oov)
        if idx % 1_000_000 == 0 && rank == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
        end
    end
    n_obs = nobs(local_stats[:fertility])
    elapsed = time() - start_time
    @info "Rank $rank has finished tokenizer stats" n_obs elapsed n_obs / elapsed

    # Reduce stats over ranks
    local_stats = OnlineStats.Series(;
        fertility=local_stats.fertility,
        nunique=local_stats.nunique,
        out_of_vocab=local_stats.out_of_vocab,
        ngrams=OnlineStats.Series(local_stats.ngrams),
    )
    tokenizer_stats = leader_reduce(merge!, local_stats)

    # Dump Initial Stats
    local stats
    if rank == 0
        @info "Tabulating stats on rank $rank"
        stats = (;
            tokenizer=tokenizer_info,
            samples=nobs(tokenizer_stats[:fertility]),
            map(value, tokenizer_stats.stats)...
        )
    end

    # Broadcast n-gram counts to all ranks, and construct n-gram models
    ngrams = rank == 0 ? value.(tokenizer_stats[:ngrams]) : nothing
    ngrams = MPI.bcast(ngrams, comm)
    ngrams = NGramModel(ngrams, tokenizer_info.vocab_size)

    # Evaluate the log probability of the validation set
    ds = setup_dm_mpi(datamodule, "train"; rank, size)
    n = length(ngrams.total)
    log_odds = zeros(n)
    for (idx, example) in enumerate(ds)
        input_ids = pyconvert(Vector{Int}, example["input_ids"])
        for n in eachindex(log_odds)
            log_odds[n] += autoregressive_kld(ngrams, input_ids; N=n)
        end
    end
    MPI.Reduce!(log_odds, +, comm)

    if rank == 0
        @info "Saving n-gram results on rank $rank"
        log_odds ./= stats.samples
        stats = (; stats..., ngram_log_odds=log_odds)
        mkpath(dirname(out_file))
        BSON.bson(out_file; stats...)
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
    rank == 0 && @info "Loaded n-gram model for $ref_name from $ref_file"

    fit_stats = Dict{Symbol, NamedTuple}(
        :tokenizer => (;
            vocab_size=pyconvert(Int, len(tok)),
            unk_token_id=pyconvert(Union{Int, Nothing}, tok.unk_token_id),
        ),
        ref_tokenizer=ref_info,
    )

    for split in ["train", "val"]
        ds = setup_dm_mpi(datamodule, split; rank, size)
        unk_token_id = pyconvert(Int, datamodule.tokenizer.unk_token_id)
        stats = map(1:length(ngram)) do _
            OnlineStats.Series(;
                moments=OnlineStats.Moments(),
                extrema=Extrema(),
            )
        end |> OnlineStats.Group
        loss = zeros(length(ngram))

        @info "rank $rank: started processing"
        for encoding in ds
            for N in 1:length(ngram)
                loss[N] += autoregressive_kld(ngram, code; N)
            end
            fit!(stats, (loss))
        end
        stats = leader_reduce(merge!, stats)
        if rank == 0
            fit_stats[Symbol(split)] = value(stats)
        end
        MPI.Barrier(comm)
    end

    if rank == 0
        mkpath(dirname(out_file))
        BSON.bson(out_file; fit_stats...)
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
    rank == 0 && @info "Loaded n-gram model for $(ref_info.name) from $ref_file"
    @assert ref_info.name == "character"

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
    for encoding in ds
        info_loss = unk_information_loss(ngram, ref_tok, tokenizer, encoding)
        fit!(stats, tuple(info_loss))
    end
    stats = leader_reduce(merge!, stats)
    if rank == 0
        @info "saving results to $output" stats
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
        BSON.bson(output; stats...)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end
