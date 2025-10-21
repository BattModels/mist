function load_generated_molecules(path)
    file = !endswith(path, ".sqlite") ? joinpath(path, "merged.sqlite") : path

    if basename(file) == "merged.sqlite" && !isfile(file)
        cmd = `uv run python -m src.cli merge-db $(dirname(file))`
        dir = joinpath(pkgdir(@__MODULE__), "..")
        run(Cmd(cmd; dir))
    end

    db = SQLite.DB(file)
    df = nothing
    try
        df = SQLite.DBInterface.execute(db, "SELECT * from molecules") |> DataFrame
    finally
        close(db)
    end
    cols = keys(JSON.parse(df[1, :props]))
    transform!(df, :props => ByRow(JSON.parse) => Symbol.(cols))
    select!(df, Not(:props))
    rename!(df, "inchi" => "inchi_key")
    return df
end

function load_all(outdir=joinpath(pkgdir(@__MODULE__), "..", "out"))
    cases = Dict{String,DataFrame}()
    for case in readdir(outdir; join=true)
        isfile(joinpath(case, "merged.sqlite")) || continue
        cases[basename(case)] = load_generated_molecules(case)
    end
    return cases
end
