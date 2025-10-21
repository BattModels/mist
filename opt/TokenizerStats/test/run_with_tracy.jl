#!/usr/bin/env -S julia --project --startup-file=no
using Tracy
@info "Waiting for tracy to connect..."
wait_for_tracy()
@info "Tracy connected!"

using TokenizerStats
using ReTestItems
@tracepoint "runtests" begin
    runtests(TokenizerStats)
end
