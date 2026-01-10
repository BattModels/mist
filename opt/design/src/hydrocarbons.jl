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

# Branched alkyl chain generators
iso_alkyl(n::Int) = n < 3 ? "C"^n : "C(C)" * "C"^(n - 2)
tert_alkyl(n::Int) = n < 4 ? "C"^n : "C(C)(C)" * "C"^(n - 3)

function branched_esters(n_total::Int)
    esters = []
    # Generate esters with different chain length splits
    for n_acyl in 1:(n_total - 1)
        n_alkoxy = n_total - n_acyl
        # Linear acyl, iso alkoxy (if n_alkoxy >= 3)
        if n_alkoxy >= 3
            push!(esters, (; smi="O=C(O" * iso_alkyl(n_alkoxy) * ")" * "C"^(n_acyl - 1),
                           branch_pattern="iso-alkoxy"))
        end
        # Iso acyl, linear alkoxy (n_acyl - 1 >= 3, so n_acyl >= 4)
        if n_acyl >= 4
            push!(esters, (; smi="O=C(O" * "C"^n_alkoxy * ")" * iso_alkyl(n_acyl - 1),
                           branch_pattern="iso-acyl"))
        end
        # Tert acyl, linear alkoxy (n_acyl - 1 >= 4, so n_acyl >= 5)
        if n_acyl >= 5
            push!(esters, (; smi="O=C(O" * "C"^n_alkoxy * ")" * tert_alkyl(n_acyl - 1),
                           branch_pattern="tert-acyl"))
        end
        # Both iso (n_acyl - 1 >= 3 and n_alkoxy >= 3, so n_acyl >= 4 and n_alkoxy >= 3)
        if n_acyl >= 4 && n_alkoxy >= 3
            push!(esters, (; smi="O=C(O" * iso_alkyl(n_alkoxy) * ")" * iso_alkyl(n_acyl - 1),
                           branch_pattern="iso-both"))
        end
    end
    return unique(x -> x.smi, esters)
end

function branched_ethers(n_total::Int)
    ethers = []
    # Generate ethers with different chain length splits
    for n_left in 1:(n_total - 1)
        n_right = n_total - n_left
        # Linear left, iso right (if n_right >= 3)
        if n_right >= 3
            push!(ethers, (; smi="C"^n_left * "O" * iso_alkyl(n_right),
                           branch_pattern="iso-right"))
        end
        # Iso left, linear right (if n_left >= 3)
        if n_left >= 3
            push!(ethers, (; smi=iso_alkyl(n_left) * "O" * "C"^n_right,
                           branch_pattern="iso-left"))
        end
        # Tert left, linear right (if n_left >= 4)
        if n_left >= 4
            push!(ethers, (; smi=tert_alkyl(n_left) * "O" * "C"^n_right,
                           branch_pattern="tert-left"))
        end
        # Linear left, tert right (if n_right >= 4)
        if n_right >= 4
            push!(ethers, (; smi="C"^n_left * "O" * tert_alkyl(n_right),
                           branch_pattern="tert-right"))
        end
        # Both iso (if both sides >= 3)
        if n_left >= 3 && n_right >= 3
            push!(ethers, (; smi=iso_alkyl(n_left) * "O" * iso_alkyl(n_right),
                           branch_pattern="iso-both"))
        end
    end
    return unique(x -> x.smi, ethers)
end

function branched_alcohols(n_total::Int)
    alcohols = []
    # Secondary alcohols - OH on carbon with iso branching (if n >= 3)
    if n_total >= 3
        push!(alcohols, (; smi="O" * iso_alkyl(n_total),
                         branch_pattern="secondary-at-O-iso"))
    end
    # Tertiary alcohols - OH on carbon with tert branching (if n >= 4)
    if n_total >= 4
        push!(alcohols, (; smi="O" * tert_alkyl(n_total),
                         branch_pattern="tertiary-at-O-tert"))
    end
    # Secondary alcohols (OH on internal carbon, linear)
    for pos in 2:(n_total - 1)
        # Linear secondary alcohol
        push!(alcohols, (; smi="C"^(pos - 1) * "C(O)" * "C"^(n_total - pos),
                         branch_pattern="secondary-linear"))

        # Tertiary alcohol with branching at OH position (both sides must exist)
        if pos >= 2 && pos <= n_total - 2
            push!(alcohols, (; smi="C"^(pos - 1) * "C(O)(C)" * "C"^(n_total - pos - 1),
                             branch_pattern="tertiary-at-O"))
        end
    end
    # Tertiary alcohols (3 distinct alkyl groups)
    if n_total >= 4
        for n_side1 in 1:(n_total - 3)
            for n_side2 in 1:(n_total - n_side1 - 2)
                n_side3 = n_total - n_side1 - n_side2 - 1
                push!(alcohols, (; smi="C"^n_side1 * "C(O)(" * "C"^n_side2 * ")" * "C"^n_side3,
                                 branch_pattern="tertiary"))
            end
        end
    end
    return unique(x -> x.smi, alcohols)
end

function branched_thiols(n_total::Int)
    thiols = []
    # Secondary thiols - SH on carbon with iso branching (if n >= 3)
    if n_total >= 3
        push!(thiols, (; smi="S" * iso_alkyl(n_total),
                       branch_pattern="secondary-at-S-iso"))
    end
    # Tertiary thiols - SH on carbon with tert branching (if n >= 4)
    if n_total >= 4
        push!(thiols, (; smi="S" * tert_alkyl(n_total),
                       branch_pattern="tertiary-at-S-tert"))
    end
    # Secondary thiols (SH on internal carbon, linear)
    for pos in 2:(n_total - 1)
        # Linear secondary thiol
        push!(thiols, (; smi="C"^(pos - 1) * "C(S)" * "C"^(n_total - pos),
                       branch_pattern="secondary-linear"))

        # Tertiary thiol with branching at SH position (both sides must exist)
        if pos >= 2 && pos <= n_total - 2
            push!(thiols, (; smi="C"^(pos - 1) * "C(S)(C)" * "C"^(n_total - pos - 1),
                           branch_pattern="tertiary-at-S"))
        end
    end
    # Tertiary thiols (3 distinct alkyl groups)
    if n_total >= 4
        for n_side1 in 1:(n_total - 3)
            for n_side2 in 1:(n_total - n_side1 - 2)
                n_side3 = n_total - n_side1 - n_side2 - 1
                push!(thiols, (; smi="C"^n_side1 * "C(S)(" * "C"^n_side2 * ")" * "C"^n_side3,
                               branch_pattern="tertiary"))
            end
        end
    end
    return unique(x -> x.smi, thiols)
end

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
        [(; type="Alkanes", smi=alkane(n)) for n in 1:n],
        [(; type="Esters", smi=ester(n)) for n in 1:n],
        [(; type="Ethers", smi=ether(n)) for n in 3:n],
        [(; type="Alcohols", smi=alcohol(n)) for n in 1:n],
        [(; type="Aldehydes", smi=aldehyde(n)) for n in 1:n],
        [(; type="Carboxylic acids", smi=carboxylic_acid(n)) for n in 2:n],
        [(; type="Alkenes", smi=alkene(n)) for n in 2:n],
        [(; type="Alkynes", smi=alkyne(n)) for n in 2:n],
        [(; type="Arenes", smi=arene(n)) for n in 0:n],
        [(; type="Thiol", smi=thiol(n)) for n in 1:n],

    ))
    n_carbon!(df)
    return df
end

function branched_fragrance_compounds(n::Int)
    df = DataFrame(vcat(
        [(; type="Branched Esters", row...) for row in branched_esters(n)],
        [(; type="Branched Ethers", row...) for row in branched_ethers(n)],
        [(; type="Branched Alcohols", row...) for row in branched_alcohols(n)],
        [(; type="Branched Thiols", row...) for row in branched_thiols(n)],
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
