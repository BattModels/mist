function estimate_datasize(N::Real, chains::ComponentArray{<:Real, 3})
    # Estimate Datasize
    A = chains[:, :, :A]
    α = chains[:, :, :α]
    B = chains[:, :, :B]
    β = chains[:, :, :β]
    inv_a = @. (α + β) / β
    inv_G = @. (β * B) / (α * A)
    D_opt = @. inv_G^inv(β) * N^(inv_a - 1)
    return D_opt
end

function estimate_lr(N::Real, chains)
    lr_0 = chains[:lr_0]
    lr_n = chains[:lr_n]
    return @. exp(lr_0 - lr_n * log(N))
end
