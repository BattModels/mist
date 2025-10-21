hoffman_scaling(N, D; A, B, α, β, E) = (A / N^α) + (B / D^β) + E
function compute_optimal_model_size(flops; A, α, B, β)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end
function compute_optimal_data_size(flops; A, α, B, β)
    G_inv = @. ((α * A) / (β * B))^(-1 / (α + β))
    b = @. α / (α + β)
    return @. G_inv * (flops / 6)^b
end
function compute_optimal_loss(C; A, α, B, β, E)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    b = @. α / (α + β)
    d = @. C / 6
    return @. E + A * (G * d^a)^(-α) + B * (inv(G) * d^b)^(-β)
end

function compute_optimal_loss(chains::AbstractChains, C::Real; p=0.95)
    loss = credible_interval(p)
    for J in CartesianIndices(axes(chains)[1:2])
        θ = chains[J.I..., :]
        (; A, α, B, β, E) = θ
        fit!(loss, compute_optimal_loss(C; A, α, B, β, E))
    end
    return OnlineStats.value(loss)
end

function compute_optimal_model_size(chains::AbstractChains, C::Real; p=0.95)
    N_opt = credible_interval(p)
    for J in CartesianIndices(axes(chains)[1:2])
        θ = chains[J.I..., :]
        (; A, α, B, β) = θ
        fit!(N_opt, compute_optimal_model_size(C; A, α, B, β))
    end
    return OnlineStats.value(N_opt)
end

function hoffman_compute_scaling(chains::AbstractChains, N::Real, C::Real; p=0.95)
    loss = credible_interval(p)
    D = C / (6 * N)
    for J in CartesianIndices(axes(chains)[1:2])
        θ = chains[J.I..., :]
        (; A, α, B, β, E) = θ
        fit!(loss, hoffman_scaling(N, D; A, α, B, β, E))
    end
    return OnlineStats.value(loss)
end

function hoffman_compute_scaling(chains::AbstractChains, N::AbstractVector, C::AbstractVector; kwargs...)
    mu = similar(chains, length(N), length(C))
    lower = similar(mu)
    upper = similar(mu)
    for J in eachindex(IndexCartesian(), mu)
        mu[J], lower[J], upper[J] = hoffman_compute_scaling(chains, N[J[1]], C[J[2]]; kwargs...)
    end
    return mu, lower, upper
end

function broadcast_compute_ci(f, chains::AbstractChains, C::AbstractVector; kwargs...)
    mu = similar(chains, length(C))
    lower = similar(mu)
    upper = similar(mu)
    for j in eachindex(IndexCartesian(), mu)
        mu[j], lower[j], upper[j] = f(chains, C[j]; kwargs...)
    end
    return mu, lower, upper
end

for fun in [:compute_optimal_loss, :compute_optimal_model_size]
    @eval $fun(chains, C::AbstractVector; p=0.95) = broadcast_compute_ci($fun, chains, C; p)
end
