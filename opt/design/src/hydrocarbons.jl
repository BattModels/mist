# Terminal Functional Groups
alkyne(n::Int) = "C#" * "C"^(n - 1)
alkene(n::Int, pos::Int=1) = "C"^pos * "=" * "C"^(n - pos)
alkane(n::Int) = "C"^n
arene(n::Int) = "c1ccccc1" * "C"^n
isoalkane(n::Int) = "C(C)" * "C"^(n - 2)
alcohol(n::Int) = "O" * "C"^n
aldehyde(n::Int) = "O=" * "C"^n
carboxylic_acid(n::Int) = "O=C(O)" * "C"^(n - 1)
nitrile(n::Int) = "N#" * "C"^n
dinitrile(n::Int) = "N#" * "C"^n * "#N"
amine(n::Int) = "N" * "C"^n
thiol(n::Int) = "S" * "C"^n
halide(n::Int, element::String) = element * "C"^n
halide(element::String) = Base.Fix2(halide, element)
ester(n::Int, m::Int=1) = "O=C(O" * "C"^m * ")" * "C"^(n - 1)
ether(n::Int, m::Int=1) = "C"^n * "O" * "C"^m

function fatty_acid(c::Int, d::Int, n::Int)
    d == 0 && return carboxylic_acid(c)
    @assert n > 0
    @assert c > n
    unstat_len = c - n
    step = fld(unstat_len, d)
    d_locs = range(; stop=unstat_len, step, length=d)
    @assert first(d_locs) > 1 "Exceeded Valance of first carbon"
    fatty_acid(c, collect(d_locs))
end
function fatty_acid(c::Int, d_locs::Vector{Int})
    stat_chains = map(alkane, diff(vcat([1], d_locs, [c])))
    return carboxylic_acid(1) * join(stat_chains, "=")
end

# Branched Functional Group
tetra_sub_alkene(n::Int) = "$("C"^n)C(=C($("C"^n))$("C"^n))$("C"^n)"
disubstituted_alkyne(n::Int) = "$("C"^n)#$("C"^n)"
triboroester(n::Int) = "O(B(O$("C"^n))O$("C"^n))$("C"^n)"
trialkylborane(n::Int) = "B($("C"^n))($("C"^n))$("C"^n)"

# Polymers
polyether(n::Int) = "COC"^Int(n / 2)

function simple_hydrocarbons(n::Int)
    df = DataFrame(vcat(
        [(; type="Alkanes", smi=alkane(n)) for n in 1:n],
        [(; type="Isoalkanes", smi=isoalkane(n)) for n in 3:n],
        [(; type="Alcohols", smi=alcohol(n)) for n in 1:n],
        [(; type="Aldehydes", smi=aldehyde(n)) for n in 1:n],
        [(; type="Nitrile", smi=nitrile(n)) for n in 1:n],
        [(; type="Dinitriles", smi=dinitrile(n)) for n in 1:n],
        [(; type="Amines", smi=amine(n)) for n in 1:n],
        [(; type="Carboxylic acids", smi=carboxylic_acid(n)) for n in 2:n],
        [(; type="Fluoroalkanes", smi=halide(n, "F")) for n in 2:n],
        [(; type="Bromoalkanes", smi=halide(n, "Br")) for n in 2:n],
        [(; type="Chloroalkanes", smi=halide(n, "Cl")) for n in 2:n],
        [(; type="Alkenes", smi=alkene(n)) for n in 2:n],
        [(; type="Alkynes", smi=alkyne(n)) for n in 2:n],
        [(; type="Arenes", smi=arene(n)) for n in 6:n],
        [(; type="Polyethers", smi=polyether(n)) for n in 2:2:n],
    ))
    n_carbon!(df)
    return df
end

function fragrance_compounds(n::Int)
    df = DataFrame(vcat(
        [(; type="Esters", smi=ester(n)) for n in 1:n],
        [(; type="Ethers", smi=ether(n)) for n in 3:n],
        [(; type="Alcohols", smi=alcohol(n)) for n in 1:n],
        [(; type="Aldehydes", smi=aldehyde(n)) for n in 1:n],
        [(; type="Carboxylic acids", smi=carboxylic_acid(n)) for n in 2:n],
        [(; type="Alkenes", smi=alkene(n)) for n in 2:n],
        [(; type="Arenes", smi=arene(n)) for n in 0:n],
        [(; type="Thiol", smi=thiol(n)) for n in 1:n],

    ))
    n_carbon!(df)
    return df
end

n_carbon!(df) = transform!(df, :smi => ByRow(smi -> count(c -> c == 'C' || c == 'c', smi)) => :n_carbon)

function saturated_fats(max_length::Int; n_max=typemax(Int), d_max=typemax(Int))
    rows = []
    for c in 3:max_length
        push!(rows, (; c, d=0, n=0, smi=carboxylic_acid(c)))
        for d in 1:min(c, d_max)
            nm = c - d - 1
            for n in 1:min(nm, n_max)
                push!(rows, (; c, d, n, smi=fatty_acid(c, d, n)))
            end
        end
    end
    df = DataFrame(rows)
    transform!(df, [:c, :d] => ByRow((c, d) -> d / (c-2)) => :saturation)
    return df
end

function alkene_sweep(n, model)
    smi = String[]
    rel_pos = Float64[]
    for i in 2:n
        for pos in 1:i-1
            push!(smi, alkene(i, pos))
            push!(rel_pos, (2pos - 1) / (2i - 2))
        end
    end
    df = predict_monte(smi, model)
    n_carbon!(df)
    df.rel_pos = rel_pos
    return df
end
