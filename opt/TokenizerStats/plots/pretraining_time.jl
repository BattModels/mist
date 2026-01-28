using SmirkPaperPlots
using DataFrames
using GLM
using RegressionTables
using CSV: CSV
using JLD2: jldopen
using TokenizerStats: load_tokenizer
using StatsBase: mean

function avg_fertility(token_usage)
    token_usage = transform(token_usage,
        :dataset => ByRow(x -> lowercase(x) in ["tmqm", "realspace"] ? x : "MoleculeNet") => :dataset
    )
    token_usage = combine(groupby(token_usage, [:tokenizer, :dataset])) do gdf
        return (;
            fertility = mean(reduce(merge, gdf.fertility))
        )
    end
    return token_usage
end

function fit_pretrained_time_models(file)
    tok_info = SmirkPaperPlots.tokenizers_info("tokenizers.json")
    df = DataFrame(CSV.File(file))
    transform!(df,
        :tokenizer => ByRow(tok -> tok_info[tok]["name"]) => :name,
        :tokenizer => ByRow(tok -> tok_info[tok]["tokenizer_class"]) => :tokenizer_class,
    )
    rename!(df, "Pretraining (sec)" => "training_time")
    @info df
    df.training_time ./= 60 * 60 # Secs to Hours
    @info df

    contrasts = Dict(
        :tokenizer => EffectsCoding(; base="smirk"),
        :tokenizer_class => EffectsCoding(; base="atomwise"),
        :encoding => EffectsCoding(; base="smiles"),
    )
    model = lm(@formula(training_time ~ tokenizer_class + encoding), df;
        contrasts
    )
    regtable(
        model;
        below_statistic = ConfInt,
        regression_statistics=[
            RegressionTables.R2,
        ],
        render = LatexTable(),
        file = "./fig/training_time.tex",
    )
    return df
end

function training_flops(l, h, s, V)
    return 96*s*l*h^2 * (1 + (s/(6*h)) + (V/(16*l*h)))
end

function query(df, column, value)
    idx = findfirst(==(value), df[!, column])
    isnothing(idx) && return missing
    return df[idx, :]
end

function flops_ratios(; l=8, h=512)
    token_usage = jldopen("tokenizer_stats.jld2", "r") do h5
        h5["token_usage"]
    end
    df_fertility = avg_fertility(token_usage)


    tokenizers = unique(token_usage.tokenizer)
    vocab_size = map(tokenizers) do tok
        try
            length(load_tokenizer(tok))
        catch
            @error "unable to load $tok"
            missing
        end
    end
    vocab_size = Dict(zip(tokenizers, vocab_size))

    df = combine(groupby(df_fertility, [:dataset])) do gdf
        gdf = select(gdf, [:tokenizer, :fertility])
        gdf.flops = map(gdf.fertility, gdf.tokenizer) do s, tok
            training_flops(l, h, s, vocab_size[tok])
        end
        select!(gdf, [:tokenizer, :flops])
        smirk_flops = query(gdf, :tokenizer, "smirk")
        smirk_flops = !ismissing(smirk_flops) ? smirk_flops.flops : smirk_flops
        gdf.flops_ratio = gdf.flops ./ smirk_flops
        select!(gdf, [:tokenizer, :flops_ratio])
        return gdf
    end
    df = unstack(df, :dataset, :flops_ratio)
    dropmissing!(df)
    sort!(df, "realspace")

    tok_info = SmirkPaperPlots.tokenizers_info("tokenizers.json")
    transform!(df,
        :tokenizer => ByRow(tok -> tok_info[tok]["name"]) => :name,
        :tokenizer => ByRow(tok -> tok_info[tok]["tokenizer_class"]) => :tokenizer_class,
    )
    write_flops_ratio_table(df)
    return df
end

function write_training_table(file; fig_dir="./fig")
    df = DataFrame(CSV.File(file))
    tok_info = SmirkPaperPlots.tokenizers_info("tokenizers.json")
    transform!(df,
        :tokenizer => ByRow(tok -> tok_info[tok]["name"]) => :name,
        :tokenizer => ByRow(tok -> tok_info[tok]["tokenizer_class"]) => :tokenizer_class,
    )
    replace!(df[!, "tokenizer_class"],
        "smirk-gpe" => "GPE",
        "spe" => "SPE",
        "bpe" => "BPE",
        "unigram" => "Unigram",
        "atomwise" => "Atom-wise"
    )
    replace!(df[!, "encoding"],
        "smiles" => "SMILES",
        "selfies" => "SELFIES",
        "smiles-canonical" => "SMILES, Canonical"
    )
    df.gpu_hours .= df[!, "Pretraining (sec)"] ./ (60 * 60)
    open(joinpath(fig_dir, "training_times.tex"), "w") do fid
        header = """
            \\begin{tabular}{lcc|c}
                \\toprule
                Tokenizer &
                Class &
                Encoding &
                Pretraining [GPU-Hours]
                \\\\
                \\midrule
            """
        write(fid, header)
        sig = x -> round(x; sigdigits=3)
        for row in eachrow(df)
            line = """
                $(row["name"]) &
                $(row["tokenizer_class"]) &
                $(row["encoding"]) &
                $(sig(row["gpu_hours"])) \\\\
            """
            write(fid, line)
        end

        footer = """
            \\bottomrule
        \\end{tabular}
        """
        write(fid, footer)
    end
    return nothing
end

function write_flops_ratio_table(df, fig_dir="./fig")
    df = select(df, All())
    replace!(df[!, "tokenizer_class"],
        "smirk-gpe" => "GPE",
        "spe" => "SPE",
        "bpe" => "BPE",
        "unigram" => "Unigram",
        "atomwise" => "Atom-wise"
    )
    open(joinpath(fig_dir, "flops_ratio.tex"), "w") do fid
        header = """
            \\begin{tabular}{ll|ccc|}
                \\toprule
                Tokenizer &
                Class &
                REALSpace &
                MoleculeNet &
                tmQM \\\\
                \\midrule
            """
        write(fid, header)
        sig = x -> round(x; sigdigits=3)
        for row in eachrow(df)
            line = """
                $(row["name"]) &
                $(row["tokenizer_class"]) &
                $(sig(row["realspace"])) &
                $(sig(row["MoleculeNet"])) &
                $(sig(row["tmQM"])) \\\\
            """
            write(fid, line)
        end

        footer = """
            \\bottomrule
        \\end{tabular}
        """
        write(fid, footer)
    end
    return nothing
end
