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
    MPI.Barrier(comm)
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
    @info "rank $rank: started processing $split"
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

function srun_usage_stats(datamodule::Py, out_file::AbstractString; tokenizer_name::AbstractString="", splits=["train"])
    size = parse(Int, ENV["SLURM_NTASKS"])
    rank = parse(Int, ENV["PMIX_RANK"])

    # Load Dataset and Tokenizer
    tokenizer = datamodule.tokenizer
    tokenizer_info = (;
        name=tokenizer_name,
        vocab_size=pyconvert(Int, length(tokenizer)),
        unk_token_id=pyconvert(Union{Int,Nothing}, tokenizer.unk_token_id),
    )
    stats = Dict{Symbol,Any}()
    splits = (length(splits) == 1 && first(splits) == "all") ? ["train", "val", "test"] : splits
    for split in splits
        tokenizer_stats = rank_usage_stats(datamodule, split; rank, size)
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
        splits = join(splits, "_")
        out_file = out_file * "_split_$(splits)_rank_$rank.bson"
        @info "Saving stats on rank $rank to $out_file"
        mkpath(dirname(out_file))
        stats[:tokenizer] = tokenizer_info
        rm(out_file; force=true)
        BSON.bson(out_file; stats...)
        chmod(out_file, 0o444)
    end

    return 0
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
        unk_token_id=pyconvert(Union{Int,Nothing}, tokenizer.unk_token_id),
    )
    stats = Dict{Symbol,Any}()
    splits = (length(splits) == 1 && first(splits) == "all") ? ["val", "train", "test"] : splits

    MPI.Barrier(comm)
    for split in splits
        rank_stats = rank_usage_stats(datamodule, split; rank, size)
        tokenizer_stats = leader_reduce(merge!, rank_stats; comm)
        if rank == 0
            @info "Saving results for $split on rank $rank"
            stats[Symbol(split)] = (;
                samples=nobs(tokenizer_stats),
                map(value, tokenizer_stats.stats)...
            )
        end
    end

    # Save stats
    if rank == 0
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
    fit_stats = Dict{Symbol,Any}(
        :tokenizer => (;
            vocab_size=pyconvert(Int, length(tok)),
            unk_token_id=pyconvert(Union{Int,Nothing}, tok.unk_token_id),
        ),
        :ref_tokenizer => ref_info,
    )

    MPI.Barrier(comm)
    for split in ["train", "val", "test"]
        ds = setup_dm_mpi(datamodule, split; rank, size)
        stats = map(1:length(ngram)) do _
            OnlineStats.Series(;
                moments=OnlineStats.Moments(),
                extrema=Extrema(),
                histogram=KHist(100),
            )
        end |> OnlineStats.Group
        stats = (; kld=deepcopy(stats), kld_per_token=deepcopy(stats))

        @info "rank $rank: started processing $split"
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
            fit_stats[Symbol(split)] = (;
                samples=nobs(stats),
                kld=map(value, stats[:kld]),
                kld_per_token=map(value, stats[:kld_per_token]),
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

@annotate function avg_information_loss(datamodule::Py, ref_file::String, output::String)
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

    MPI.Barrier(comm)
    smi_column = pyconvert(String, datamodule.smi_column)
    start_time = time()
    for (idx, encoding) in enumerate(ds)
        info_loss = unk_information_loss(ngram, ref_tok, tokenizer, encoding; smi_column)
        fit!(stats, tuple(info_loss))
        if idx % 10 == 0 && rank == 0
            elapsed = time() - start_time
            @info "rank $rank on molecule $idx" idx elapsed idx / elapsed
        end
    end
    stats = leader_reduce(merge!, stats; comm)
    if rank == 0
        @info "saving results to $output"
        tok = datamodule.tokenizer
        stats = (;
            tokenizer=(;
                vocab_size=pyconvert(Int, length(tok)),
                unk_token_id=pyconvert(Union{Int,Nothing}, tok.unk_token_id),
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

merge_usage_stats(files::Vector{String}) = mapreduce(BSON.load, _merge_rank_usage_stats, files)
function merge_usage_stats(dir::String; split::AbstractString="train", glob::Regex=Regex(".*_split_$(split)_rank_\\d+\\.bson"))
    files = find(dir, glob)
    return merge_usage_stats(files)
end

function _merge_rank_usage_stats(a::Dict, b::Dict)
    @assert a[:tokenizer] == b[:tokenizer] "All files must use the same tokenizer"
    for k in [:train, :val, :test]
        if !haskey(a, k) && haskey(b, k)
            a[k] = b[k] # b has a key, but not a, so just copy

        elseif haskey(a, k) && haskey(b, k)
            # Both have a key, so merge
            a[k] = _merge_usage_stats(a[k], b[k])
        end
    end
    return a
end

function _merge_usage_stats(a::NamedTuple, b::NamedTuple)
    @assert Set(keys(a)) == Set(keys(b)) == Set([:samples, keys(tracked_stats())...])
    samples = a.samples + b.samples
    nunique = mergewith(+, a.nunique, b.nunique)
    fertility = mergewith(+, a.fertility, b.fertility)
    out_of_vocab = a.out_of_vocab + b.out_of_vocab
    ngrams = map(1:5) do n
        ang = a.ngrams[n]
        bng = b.ngrams[n]
        @assert keytype(ang) <: NTuple{n}
        @assert keytype(bng) <: NTuple{n}
        mergewith(+, compact_ngrams(ang), compact_ngrams(bng))
    end
    out = (; samples, nunique, fertility, out_of_vocab, ngrams)
    @assert Set(keys(out)) == Set([:samples, keys(tracked_stats())...])
    return out
end

function compact_ngrams(ngrams::AbstractDict)
    keytype(ngrams) == UInt16 && valuetype(ngrams) == Float32 && return ngrams
    map(collect(pairs(ngrams))) do (k, v)
        k = UInt16.(k)
        k => Float32(v)
    end |> Dict
end
