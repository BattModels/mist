function batch_score(model::Py, smiles::Vector{String}; batch_size=32)
    _eval(batch) =  pyconvert(Vector{Float64}, model.score(PyList(batch)))
    ds = Iterators.partition(smiles, batch_size)
    out = Iterators.map(_eval, ds)
    return collect(Iterators.flatten(out))
end
