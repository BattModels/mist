#!/usr/bin/env -S julia --project --threads=4 --startup-file=no
using TokenizerStats: main
using Tracy: wait_for_tracy

if haskey(ENV, "TRACY_ENABLE") && get(ENV, "PMIX_RANK", -1) == 0
    @info "Waiting for tracy to connect..."
    wait_for_tracy()
    @info "Connected!"
end

exit(main(ARGS))
