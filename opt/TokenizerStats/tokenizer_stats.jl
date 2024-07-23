#!/usr/bin/env -S julia --project --startup-file=no
using TokenizerStats: tabulate_dataset

function main(args::Vector{String})
    @assert length(args) >= 2
    ds_path = args[1]
    tok_name = args[2]
    out_file = length(args) == 3 ? args[3] : "stats.json"
    tabulate_dataset(ds_path, tok_name, out_file)
end

!isinteractive() && main(ARGS)
