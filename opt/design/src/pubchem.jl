function pubchem_from_jsonl(file::String)
    rows = []
    open(file) do f
        while !eof(f)
            mol = JSON.parse(readline(f))
            min_lowdin = Inf
            lowdin = Float64[]
            mulliken = Float64[]
            for atom in mol["atoms"]
                push!(lowdin, atom["properties"]["partial-charge-lowdin"])
                push!(lowdin, atom["properties"]["hs_partial-charge-lowdin"])
                push!(mulliken, atom["properties"]["partial-charge-mulliken"])
                push!(mulliken, atom["properties"]["hs_partial-charge-mulliken"])
            end
            push!(rows, (;
                smi=mol["smi"],
                min_lowdin=minimum(lowdin),
                min_mulliken=minimum(mulliken),
                range_lowdin=-(extrema(lowdin)...),
                range_mulliken=-(extrema(mulliken)...),
            ))
        end
    end
    return DataFrame(rows)
end
