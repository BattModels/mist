using DataFrames
using JSON: JSON
using Format: format
using TokenizerStats: TokenizerStats
using SmirkPaperPlots: SmirkPaperPlots, savefig

stats_dir = joinpath(pkgdir(TokenizerStats), "stats")

function oov_table(stats_dir)
    tokenizers_info = SmirkPaperPlots.tokenizers_info(stats_dir)
    rows = []
    for (tok_name, tok_info) in tokenizers_info
        oov_stat_file = joinpath(stats_dir, tok_info["name_or_path"], "oov.json")
        isfile(oov_stat_file) || continue
        oov_stats = JSON.parsefile(oov_stat_file)
        tok_coverage = Dict()
        for (group, gs) in oov_stats
            total = gs["nobs"] + gs["failed_encode"]
            covered = total - gs["oov"] -gs["failed_encode"]

            if startswith(group, "MoleculeNet")
                if !haskey(tok_coverage, "MoleculeNet")
                    tok_coverage["MoleculeNet"] = (0, 0)
                end
                pcov, ptot = tok_coverage["MoleculeNet"]
                tok_coverage["MoleculeNet"] = (pcov + covered, ptot + total)
            else
                tok_coverage[group] = (covered, total)
            end
        end
        group_stats = Dict(Symbol(k) => c/t for (k, (c,t)) in tok_coverage)
        !isempty(group_stats) || continue
        haskey(group_stats, :tmQM) || continue

        class = tok_info["tokenizer_class"]
        class = replace(class,
            "smirk-gpe" => "GPE",
            "spe" => "SPE",
            "bpe" => "BPE",
            "unigram" => "Unigram",
            "atomwise" => "Atom-wise"
        )

        push!(rows, (;
            name=tok_info["name"],
            name_or_path=tok_info["name_or_path"],
            class,
            cite=get(tok_info, "cite", nothing),
            source=tok_info["source"],
            encoding=tok_info["encoding"],
            tokenizer_class=tok_info["tokenizer_class"],
            group_stats...
        ))
    end
    return DataFrame(rows)
end

function write_table(df)
    tbl_name = map(eachrow(df)) do tok
        if isnothing(tok["cite"])
            if tok["name_or_path"] != "character"
                return "\\gh{BattModels/smirk} $(tok["name"]) (ours)"
            else
                return "\\dd{} $(tok["name"]) (ours)"
            end
        else
            src = tok["source"] == "huggingface" ? "hf" : "gh"
            name_or_path = tok["name_or_path"]
            name_or_path = name_or_path == "SmilesPE/SPE_ChEMBL" ? "XinhaoLi74/SmilesPE" : name_or_path # Replace with Github repo name
            name_or_path = replace(name_or_path, "_" => "\\_")
            name = replace(tok["name"], "_" => "\\_")
            return "\\$src{$name_or_path} $name\\cite{$(join(tok["cite"], ","))}"
        end
    end

    df_out = select(df, All())
    df_out.tbl_name = tbl_name

    # Format for publication
    replace!(df_out[!, "encoding"], "selfies" => "SELFIES", "smiles" => "SMILES")

    fig_dir = joinpath(pkgdir(TokenizerStats), "fig")
    mkpath(fig_dir)
    open(joinpath(fig_dir, "oov_summary.tex"), "w") do fid
        header = """
            \\begin{tabular}{lll|ccccccccc}
                Tokenizer &
                Encoding &
                Class &
                Elements &
                Bond &
                Isotopes &
                Carbon Rings &
                Ions &
                Chirality &
                Charged, Chiral Isotopes &
                MoleculeNet &
                tmQM \\\\ \\hline
            """
        write(fid, header)

        for row in eachrow(df_out)
            line = """
                $(row["tbl_name"]) &
                $(row["encoding"]) &
                $(row["class"]) &
                $(format("{:.2f}\\%", 100 * row["elements"])) &
                $(format("{:.2f}\\%", 100 * row["bonds"])) &
                $(format("{:.2f}\\%", 100 * row["isotopes"])) &
                $(format("{:.2f}\\%", 100 * row["rings"])) &
                $(format("{:.2f}\\%", 100 * row["charged_elements"])) &
                $(format("{:.2f}\\%", 100 * row["chiral_elements"])) &
                $(format("{:.2f}\\%", 100 * row["charged_chiral_isotopes"])) &
                $(format("{:.2f}\\%", 100 * row["MoleculeNet"])) &
                $(format("{:.2f}\\%", 100 * row["tmQM"])) \\\\
            """
            write(fid, line)
        end

        footer = """
            \\end{tabular}
        """
        write(fid, footer)
    end

    return df_out
end
