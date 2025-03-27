alkyne(n::Int) = "C#" * "C"^(n - 1)
alkene(n::Int) = "C=" * "C"^(n - 1)
alkene(n::Int, pos::Int) = "C"^pos * "=" * "C"^(n - pos)
alkane(n::Int) = "C"^n
arene(n::Int) = "c1ccccc1" * "C"^n
isoalkane(n::Int) = "C(C)" * "C"^(n - 2)
alcohol(n::Int) = "O" * "C"^n
aldehyde(n::Int) = "O=" * "C"^n
nitrile(n::Int) = "N#" * "C"^n
dinitrile(n::Int) = "N#" * "C"^n * "#N"
amine(n::Int) = "N" * "C"^n
carboxylic_acid(n::Int) = "C(=O)" * "C"^(n - 1)
halide(n::Int, element::String) = element * "C"^n
halide(element::String) = Base.Fix2(halide, element)
tetra_sub_alkene(n::Int) = "$("C"^n)C(=C($("C"^n))$("C"^n))$("C"^n)"
disubstituted_alkyne(n::Int) = "$("C"^n)#$("C"^n)"
triboroester(n::Int) = "O(B(O$("C"^n))O$("C"^n))$("C"^n)"
trialkylborane(n::Int) = "B($("C"^n))($("C"^n))$("C"^n)"
ether(n::Int) = "COC"^Int(n / 2)
ester(n::Int) = "C"^n * "C(=O)O" * "C"^n

function simple_hydrocarbons(n::Int)
    df = DataFrame(vcat(
        [(; type="Alkanes", smi=alkane(n)) for n in 1:n],
        [(; type="Isoalkanes", smi=isoalkane(n)) for n in 3:n],
        [(; type="Alcohols", smi=alcohol(n)) for n in 1:n],
        [(; type="Nitrile", smi=nitrile(n)) for n in 1:n],
        [(; type="Dinitriles", smi=dinitrile(n)) for n in 1:n],
        [(; type="Amines", smi=amine(n)) for n in 1:n],
        [(; type="Carboxylic Acids", smi=carboxylic_acid(n)) for n in 2:n],
        [(; type="Fluoroalkanes", smi=halide(n, "F")) for n in 2:n],
        [(; type="Bromoalkanes", smi=halide(n, "Br")) for n in 2:n],
        [(; type="Chloroalkanes", smi=halide(n, "Cl")) for n in 2:n],
        [(; type="Alkenes", smi=tetra_sub_alkene(n)) for n in 1:cld(n, 4)],
        [(; type="Alkynes", smi=disubstituted_alkyne(n)) for n in 1:cld(n, 2)],
        [(; type="Ether", smi=ether(n)) for n in 2:2:n],
        [(; type="Arene", smi=arene(n)) for n in 6:n],
        [(; type="Ester", smi=ester(n)) for n in 1:cld(n, 2)],
    ))
    n_carbon!(df)
    return df
end

n_carbon!(df) = transform!(df, :smi => ByRow(smi -> count(==('C'), smi)) => :n_carbon)

function alkene_sweep(n, model)
    smi = String[]
    rel_pos = Float64[]
    for i in 2:n
        for pos in 1:i-1
            push!(smi, alkene(i, pos))
            push!(rel_pos, (2pos-1)/(2i-2))
        end
    end
    df = predict_monte(smi, model)
    n_carbon!(df)
    df.rel_pos = rel_pos
    return df
end
