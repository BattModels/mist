struct FeatureStats{TD,M,I}
    feature_activations::Vector{TD}
    positive_proxy::Vector{M}
    negative_proxy::Vector{M}
    proxy_count::Vector{I}
    n::Ref{I}
end

function FeatureStats(n_feature::Integer, n_proxy::Integer)
    f_act = map(_ -> Series(; hist=KHist(100), var=Variance()), 1:n_feature)
    pos_proxy = map(_ -> ElementwiseVariance(Float32, n_feature), 1:n_proxy)
    proxy_count = zeros(UInt64, n_proxy)
    neg_proxy = map(_ -> ElementwiseVariance(Float32, n_feature), 1:n_proxy)
    count = zero(UInt64)
    TD = eltype(f_act)
    M = eltype(pos_proxy)
    I = eltype(proxy_count)
    FeatureStats{TD,M,I}(f_act, pos_proxy, neg_proxy, proxy_count, count)
end


"""
    fit!(stats, f_act, mask, proxy_act)

Update feature statistics `stats` with the provided activations for a single observation.

`f_act`: activations of the features (seq_len, n_features)
`mask`: mask of the observations (seq_len,)
`proxy_act`: activations of the proxies (seq_len, n_proxies)

"""
function OnlineStatsBase.fit!(stats::FeatureStats, f_act::AbstractMatrix, proxy_act::AbstractMatrix, mask::AbstractVector,)
    stats.n[] += 1

    # Update f activation statistics
    for i in axes(f_act, 2)
        f_act_masked = f_act[mask, i]
        OnlineStats.fit!(stats.feature_activations[i], Float64.(f_act_masked))
    end

    # Update proxy activation statistics
    for fdx in axes(proxy_act, 2)
        for tdx in axes(proxy_act, 1)
            mask[tdx] || continue # Skip if token is masked
            fa = @view f_act[tdx, :]
            proxy_active = false
            # Maintain separate statistics for positive and negative proxies
            pa = proxy_act[tdx, fdx]
            if pa
                proxy_active = true
                OnlineStats.fit!(stats.positive_proxy[fdx], fa)
            else
                OnlineStats.fit!(stats.negative_proxy[fdx], fa)
            end
            stats.proxy_count[fdx] += proxy_active
        end
    end

    return nothing
end

function tabulate_features(pipeline::Py; batch_size::Integer=64)
    stats = FeatureStats(512, 32)
    for batch in pipeline.iter("val"; batch_size)
        f_act = pyconvert(Array, batch["feature_activations"])
        mask = pyconvert(Matrix, batch["attention_mask"].T)
        proxy_act = pyconvert.(Matrix{Bool}, batch["proxy_activations"])
        for idx in axes(f_act, 1)
            # Update activation statistics
            fit!(stats, selectdim(f_act, 1, idx), selectdim(mask, 2, idx), proxy_act[idx, :])
        end
    end
    return stats
end
