hoffman_scaling(N, D; A, B, α, β, E) = @. (A / N^α) + (B / D^β) + E
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

function compute_optimal_loss(chains::AbstractArray{<:Real,3}, C::Vector; p=0.95)
    μ = similar(chains, length(C))
    band = similar(chains, 2, length(C))
    A = selectdim(chains, 3, :A)
    α = selectdim(chains, 3, :α)
    B = selectdim(chains, 3, :B)
    β = selectdim(chains, 3, :β)
    E = selectdim(chains, 3, :E)
    p = (1 - p) / 2 # Get lower quantile for the prediction interval
    for (i, c) in enumerate(C)
        loss = vec(compute_optimal_loss(c; A, α, B, β, E))
        band[:, i] .= quantile(loss, (p, 1 - p))
        μ[i] = mean(loss)
    end
    return μ, band
end

function compute_optimal_model_size(chains::AbstractArray{<:Real,3}, C::Vector; p=0.95)
    μ = similar(chains, length(C))
    band = similar(chains, 2, length(C))
    N = similar(chains, size(chains, 1), size(chains, 2))
    p = (1 - p) / 2 # Get lower quantile for the prediction interval
    for (i, c) in enumerate(C)
        for I in eachindex(IndexCartesian(), N)
            θ = @view chains[I.I..., :]
            N[I] = compute_optimal_model_size(c; A=θ.A, α=θ.α, B = θ.B, β=θ.β)
        end
        μ[i] = mean(N)
        band[:, i] .= quantile(N, (p, 1 - p))
    end
    return μ, band
end

function hoffman_compute_scaling(chains::AbstractArray{<:Real,3}, N::Vector, C::Vector; p=0.95)
    μ = similar(chains, length(N), length(C))
    band = similar(chains, 2, length(N), length(C))
    p = (1 - p) / 2 # Get lower quantile for the prediction interval
    loss = similar(chains, size(chains, 1), size(chains, 2))
    for I in eachindex(IndexCartesian(), μ)
        n = N[I[1]]
        d = C[I[2]] / (6n)
        for J  in eachindex(IndexCartesian(), loss)
            θ = @view chains[J.I..., :]
            loss[J] = hoffman_scaling(n, d; θ...)
        end
        band[:, I.I...] .= quantile(vec(loss), (p, 1 - p))
        μ[I] = mean(loss)
    end
    return μ, band
end

