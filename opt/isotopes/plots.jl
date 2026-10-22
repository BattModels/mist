#!/usr/bin/env -S julia --startup-file=no
# Render the isotope panel using the checked-in projection settings.

using CSV
using DataFrames
using InterpEmb
using JSON
using Makie: with_theme
using MISTStyle

function (@main)(args)
    iseven(length(args)) || error("Expected --embeddings FILE --protocol FILE --output-dir DIRECTORY")
    options = Dict{String,String}()
    for index in 1:2:length(args)
        args[index] in ["--embeddings", "--protocol", "--output-dir"] || error("Unknown option $(args[index])")
        options[args[index][3:end]] = args[index + 1]
    end
    for key in ["embeddings", "protocol", "output-dir"]
        haskey(options, key) || error("Missing --$key")
    end
    for key in ["embeddings", "protocol"]
        isfile(options[key]) || error("Missing isotope $key: $(options[key]). See opt/isotopes/README.md.")
    end
    protocol = JSON.parsefile(options["protocol"])
    required = ["pooling", "projection_columns", "n_neighbors", "min_dist", "metric", "seed", "tol_offset", "tol_scale", "umap_kwargs"]
    all(haskey(protocol, key) for key in required) || error("Incomplete isotope protocol: required $(join(required, ", "))")
    df = DataFrame(CSV.File(options["embeddings"]))
    nrow(df) > Int(protocol["n_neighbors"]) || error("The isotope cohort must contain more rows than n_neighbors")
    columns = protocol["projection_columns"]
    if columns == "embeddings"
        columns = filter(name -> startswith(name, "embedding_"), names(df))
    elseif columns == "legacy_numeric"
        columns = filter(!=("smiles"), names(df))
    end
    columns isa AbstractVector && !isempty(columns) || error("Supply a nonempty ordered projection column list or embeddings/legacy_numeric")
    protocol["metric"] in ["manhattan", "cosine", "euclidean"] || error("Unknown projection metric")
    all(name in names(df) for name in columns) || error("Projection columns are missing from the embedding CSV")
    values = Matrix(df[:, String.(columns)])
    all(value -> value isa Real && isfinite(value), values) || error("Projection columns contain nonfinite or nonnumeric values")
    kwargs = (; (Symbol(key) => value for (key, value) in protocol["umap_kwargs"])...)
    output = options["output-dir"]
    mkpath(output)
    figure = with_theme(MISTStyle.theme()) do
        InterpEmb.figure_isotopes_umap(
            Int(protocol["n_neighbors"]), protocol["min_dist"], Symbol(protocol["metric"]), Int(protocol["seed"]);
            csv_path=options["embeddings"], embedding_columns=columns,
            coordinates_path=joinpath(output, "isotope-projection.csv"),
            tol_offset=protocol["tol_offset"], tol_scale=protocol["tol_scale"], umap_kwargs=kwargs)
    end
    MISTStyle.savefig("isotope_clusters", figure; fig_dir=output)
end
