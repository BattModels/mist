function get_similarity(smiles1::AbstractString, smiles2::AbstractString, similarity_data::Dict)

    key1 = "$(smiles1)_$(smiles2)"
    key2 = "$(smiles2)_$(smiles1)"

    if haskey(similarity_data, key1)
        return similarity_data[key1]
    elseif haskey(similarity_data, key2)
        return similarity_data[key2]
    else
        return nothing
    end
end

function weighted_ternary_similarity(solvent_list::Vector{String}, compositions::Vector{Float64}, similarity_data::Dict)
    compositions = compositions ./ sum(compositions)
    weighted_sum = 0.0
    total_weight = 0.0

    for i in 1:length(solvent_list)
        for j in (i+1):length(solvent_list)
            sim = get_similarity(solvent_list[i], solvent_list[j], similarity_data::Dict)
            weight = compositions[i] + compositions[j]
            weighted_sum += weight * sim
            total_weight += weight
        end
    end

    return weighted_sum / total_weight
end
