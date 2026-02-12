function get_attention_maps(model_path:: AbstractString, smiles:: AbstractString)
    py_result = pymodel[].get_attention_maps(Py(model_path),  Py(smiles))

    # Check if loading failed
    if pyisinstance(py_result["tokenizer_name"], pybuiltins.type(pybuiltins.None))
        return (tokenizer_name=nothing, tokens=nothing, attention_maps=nothing)
    end

    attention_maps = [
        pyconvert(Array, attn.cpu().numpy())
        for attn in py_result["attention_maps"]
    ]

    return (
        tokenizer_name = pyconvert(String, py_result["tokenizer_name"]),
        tokens = pyconvert(Vector{String}, py_result["tokens"]),
        attention_maps = attention_maps
    )
end


function pool_pick_attention(attentions, layer=nothing, head=nothing, pool=nothing)
    if isnothing(layer) && isnothing(head)
        if pool in ["mean", "avg"]
            # Pool across all layers and heads
            attention = mean([mean(a[1, :, :, :], dims=1)[1, :, :] for a in attentions])
        elseif pool == "max"
            attention = mean([maximum(a[1, :, :, :], dims=1)[1, :, :] for a in attentions])
        end
    elseif !isnothing(layer)
        if layer isa Vector
            # Pool across specified layer range
            layer_range = layer[1]:layer[end]
            if pool in ["mean", "avg"]
                attention = mean([mean(a[1, :, :, :], dims=1)[1, :, :] for a in attentions[layer_range]])
            elseif pool == "max"
                attention = mean([maximum(a[1, :, :, :], dims=1)[1, :, :] for a in attentions[layer_range]])
            end
        else
            # Single layer
            if pool in ["mean", "avg"]
                attention = mean(attentions[layer][1, :, :, :], dims=1)[1, :, :]
            elseif pool == "max"
                attention = maximum(attentions[layer][1, :, :, :], dims=1)[1, :, :]
            else
                attention = attentions[layer][1, :, :, :]
            end
        end
    elseif !isnothing(head)
        # Specific head across all layers
        if pool in ["mean", "avg"]
            attention = mean([a[1, head, :, :] for a in attentions])
        elseif pool == "max"
            attention = maximum([a[1, head, :, :] for a in attentions])
        else
            attention = attentions[1][1, head, :, :]  # First layer, specified head
        end
    else
        # Specific layer and head
        attention = attentions[layer][1, head, :, :]
    end

    return attention
end
