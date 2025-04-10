function pubchem_from_jsonl(file::String)
    rows = []
    span(x) = maximum(x) - minimum(x)
    open(file) do f
        while !eof(f)
            mol = JSON.parse(readline(f))
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
                max_lowdin=maximum(lowdin),
                max_mulliken=maximum(mulliken),
                range_lowdin=span(lowdin),
                range_mulliken=span(mulliken),
            ))
        end
    end
    return DataFrame(rows)
end
