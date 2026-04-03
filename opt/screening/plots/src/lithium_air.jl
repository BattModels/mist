function load_lithium_air_reference(csv_path::String, models_dir::String, equations_module)
    cache_path = replace(csv_path, ".csv" => "_cached.csv")
    if isfile(cache_path)
        @info "Loading cached reference molecules from $cache_path"
        df_ref = DataFrame(CSV.File(cache_path))
        df_ref.smi = String.(df_ref.smi)
        @info "Loaded $(nrow(df_ref)) reference molecules from cache"
        return df_ref
    end

    @info "Cache not found, computing reference molecule properties..."
    df_ref = DataFrame(CSV.File(csv_path))
    df_ref.smi = String.(df_ref.smi)
    @info "Loaded $(nrow(df_ref)) reference molecules from CSV"

    torch = pyimport("torch")
    cuda_available = pyconvert(Bool, torch.cuda.is_available())

    mist_bp = load_mist_pretrained("mist-models/mist-26.9M-b302p09x-bp")
    df_pred = predict_mist(mist_bp, df_ref.smi)
    df_ref.bp = df_pred.bp

    mist_mp = load_mist_pretrained("mist-models/mist-26.9M-y3ge5pf9-mp")
    df_pred = predict_mist(mist_mp, df_ref.smi)
    df_ref.mp = df_pred.mp

    mist_pka = load_mist_pretrained("mist-models/mist-28M-6zlgl2qn-pKa")
    df_pred = predict_mist(mist_pka, df_ref.smi)
    df_ref.pKa_DMSO = df_pred.DMSO_pKa

    MISTMultiTask = pyimport("electrolyte_fm.models.prod_finetune").MISTMultiTask
    mist_multi = MISTMultiTask.from_pretrained(
        "../../../hf-models/mist-multi-28.2M-solvent-properties",
        trust_remote_code=true
    ).eval()
    preds = mist_multi.predict(PyList(df_ref.smi))
    df_ref.beta = pyconvert(Vector{Float64}, preds["beta"]["value"].numpy())
    @info "Beta values range: $(extrema(df_ref.beta))"

    mist_etn = load_mist_pretrained("mist-models/mist-27.1M-1gcxtg8y-ETN")
    df_pred = predict_mist(mist_etn, df_ref.smi; batch_size=16, verbose=true)
    df_ref.ETN = df_pred.ETN

    mist_homo = load_mist_pretrained("mist-models/mist-26.9M-kkgx0omx-qm9")
    df_pred = predict_mist(mist_homo, df_ref.smi)
    df_ref.homo = df_pred.homo .* HARTREE_TO_EV  # Convert from Hartree to eV

    beta_tensor = torch.tensor(df_ref.beta)
    etn_tensor = torch.tensor(df_ref.ETN)

    delta_G_tensor = equations_module.solution_mediated_reaction(beta_tensor, etn_tensor)
    df_ref.delta_G_HA_sol = pyconvert(Vector{Float64}, delta_G_tensor.numpy())

    @info "Equation-based properties computed"

    @info "Saving to cache: $cache_path"
    CSV.write(cache_path, df_ref)

    return df_ref
end

function label_lithium_air_pareto!(df; properties=[:mp, :bp, :pKa_DMSO, :delta_G_HA_sol, :homo])
    # Optimization objectives (all transformed to minimization):
    mp = df[!, :mp]                          # minimize mp
    bp = -df[!, :bp]                         # maximize bp (minimize -bp)
    pKa_DMSO = -df[!, :pKa_DMSO]             # maximize pKa_DMSO
    delta_G_HA_sol = df[!, :delta_G_HA_sol]  # minimize delta_G_HA_sol
    homo = df[!, :homo]                      # minimize homo

    candidates = map(vcat, mp, bp, pKa_DMSO, delta_G_HA_sol, homo)
    front = Metaheuristics.get_non_dominated_solutions(candidates)
    nidx = findall(in(front), candidates)
    df.dominated .= true
    df.dominated[nidx] .= false
    sort!(df, :dominated; rev=true)
    return df
end

function save_lithium_air_pareto_front(df_mol)
    df_front = deepcopy(df_mol)

    # Convert HOMO from Hartree to eV
    df_front.homo = df_front.homo .* HARTREE_TO_EV

    label_lithium_air_pareto!(df_front)
    subset!(df_front, :dominated => ByRow(!))
    sort!(df_front, [:delta_G_HA_sol, :pKa_DMSO, :homo])

    df_front.id = 1:nrow(df_front)
    columns = [
        "id" => "#",
        "mp" => LatexCell("mp (\$^\\circ\$C)"),
        "bp" => LatexCell("bp (\$^\\circ\$C)"),
        "pKa_DMSO" => LatexCell(raw"\(\text{pK}_a\) (DMSO)"),
        "delta_G_HA_sol" => LatexCell(raw"\(\Delta G_{\text{sol}}\) (eV)"),
        "homo" => LatexCell("HOMO (eV)"),
    ]

    fmt_lr(v, i, j) = format("{:.2f}", v)

    table = pretty_table(String, df_front[!, first.(columns)];
        # header=last.(columns),
        formatters=[fmt_lr],
        alignment=[:c for _ in columns],
        backend=:latex,
    )
    return table, df_front
end

function plot_lithium_air_pareto(df, df_ref=nothing)
    f = Figure(; size=(5inch, 2inch), figure_padding=(8, 8, 8, 8))

    # Make a copy to avoid modifying the original dataframe
    df = deepcopy(df)
    df.homo = df.homo .* HARTREE_TO_EV

    label_lithium_air_pareto!(df)
    label_lithium_air_pareto!(df_ref)

    df_pareto = df[.!df.dominated, :]
    pareto_kwargs = (linewidth=1.5pt, linestyle=:solid, alpha=0.7)

    ax1 = Axis(f[1, 1]; xlabel="Melt (°C)", ylabel="Boil (°C)", limits=((nothing, 25), (75, 300)))
    scatter_samples!(ax1, df.mp, df.bp, df.dominated)
    stairs!(ax1, get_pareto_front(df_ref.mp[.!df_ref.dominated], df_ref.bp[.!df_ref.dominated]; quad=:lt, ax=ax1);
        color=MISTStyle.UM_COLORS.maize, pareto_kwargs...)
    stairs!(ax1, get_pareto_front(df.mp[.!df.dominated], df.bp[.!df.dominated]; quad=:lt, ax=ax1);
        color=MISTStyle.UM_COLORS.blue, pareto_kwargs...)
    sublabel!(f[1, 1, TopLeft()], "a"; left=14pt)

    ax2 = Axis(f[1, 2]; xlabel=L"\text{pK}_a \text{ (DMSO)}", ylabel=L"\Delta\;G^{HA}_{sol} \text{ (eV)}",
            limits = ((30, 70), (-3.8, -2.4)))
    scatter_samples!(ax2, df.pKa_DMSO, df.delta_G_HA_sol, df.dominated)
    # Use quad=:rt to maximize pKa_DMSO (x) and minimize delta_G_HA_sol (y)
    stairs!(ax2, get_pareto_front(df_ref.pKa_DMSO[.!df_ref.dominated], df_ref.delta_G_HA_sol[.!df_ref.dominated]; quad=:rt, ax=ax2);
        color=MISTStyle.UM_COLORS.maize, pareto_kwargs...)
    stairs!(ax2, get_pareto_front(df.pKa_DMSO[.!df.dominated], df.delta_G_HA_sol[.!df.dominated]; quad=:rt, ax=ax2);
        color=MISTStyle.UM_COLORS.blue, pareto_kwargs...)
    sublabel!(f[1, 2, TopLeft()], "b"; left=14pt)

    ax3 = Axis(f[1, 3]; xlabel=L"\Delta\;G^{HA}_{sol} \text{ (eV)}", ylabel="HOMO (eV)", limits = ( (-3.8, -2.4), (-9., -7.01)))
    h, h_front = scatter_samples!(ax3, df.delta_G_HA_sol, df.homo, df.dominated)
    # Use quad=:ll to minimize both delta_G_HA_sol and homo
    h_ref_front = stairs!(ax3, get_pareto_front(df_ref.delta_G_HA_sol[.!df_ref.dominated], df_ref.homo[.!df_ref.dominated]; quad=:ll, ax=ax3);
        color=MISTStyle.UM_COLORS.maize, label="Ref. Pareto Front", pareto_kwargs...)
    h_gen_front = stairs!(ax3, get_pareto_front(df.delta_G_HA_sol[.!df.dominated], df.homo[.!df.dominated]; quad=:ll, ax=ax3);
        color=MISTStyle.UM_COLORS.blue, label="Generated Pareto Front", pareto_kwargs...)
    sublabel!(f[1, 3, TopLeft()], "c"; left=8pt)

    h.label = "Generated"
    h_front.label = "On Pareto Front, Generated"
    elems = [h, h_front,
        LineElement(; label=h_ref_front.label, linestyle=:solid, color=h_ref_front.color, linewidth=h_ref_front.linewidth),
        LineElement(; label=h_gen_front.label, linestyle=:solid, color=h_gen_front.color, linewidth=h_gen_front.linewidth)
    ]
    Legend(f[2, :], elems, MISTStyle.label.(elems); nbanks = 4, orientation=:vertical, tellheight=true, tellwidth=false)
    return f
end

function plot_lithium_air_homo_colored(df, df_ref=nothing)
    f = Figure(; size=(2.1inch, 2.1inch), figure_padding=(2, 2, 2, 5))

    df = deepcopy(df)
    df.homo = df.homo .* HARTREE_TO_EV

    ax = Axis(f[1, 1];
        xlabel=L"\Delta\;G^{HA}_{sol} \text{ (eV)}",
        ylabel=L"\text{pK}_a \text{ (DMSO)}",
    )

    vlines!(ax, -0.35; color=:gray, linestyle=:dash, linewidth=1.5)
    hlines!(ax, 31.3; color=:gray, linestyle=:dash, linewidth=1.5)

    text!(ax, -0.35, 35; text=L"\Delta G = -0.35 \text{ eV}",
            fontsize = 6,
          color=:gray, rotation=π/2, align=(:center, :bottom))
    text!(ax, -2.0, 31.3; text=L"\text{pK}_a = 31.3",
            fontsize = 6,
          color=:gray, align=(:right, :bottom))

    sc_gen = scatter!(ax, df.delta_G_HA_sol, df.pKa_DMSO;
        color=df.homo,
        colormap=:viridis,
        alpha=0.7,
        label="Generated"
    )

    sc_ref = scatter!(ax, df_ref.delta_G_HA_sol, df_ref.pKa_DMSO;
        color=df_ref.homo,
        colormap=:viridis,
        marker=:xcross,
        alpha=0.8,
        label="Reference"
    )

    Colorbar(f[1, 2], sc_gen; label="HOMO (eV)")
    Legend(f[2, 1], ax; orientation=:horizontal, tellheight=true, tellwidth=false)

    return f
end

function plot_lithium_air_homo_pka(df, df_ref=nothing)
    f = Figure(; size=(2.1inch, 2.1inch), figure_padding=(5, 5, 5, 5))

    df.homo = df.homo .* HARTREE_TO_EV

    ax = Axis(f[1, 1];
        xlabel="HOMO (eV)",
        ylabel=L"\text{pK}_a \text{ (DMSO)}",
        limits = ((-10, -3), (1, 65))
    )

    vlines!(ax, -7.01; color=:gray, linestyle=:dash, linewidth=1.5)
    hlines!(ax, 30; color=:gray, linestyle=:dash, linewidth=1.5)

    text!(ax, -7.01, 25; text="HOMO = -7.01 eV",
          fontsize = 6,
          color=:gray, rotation=π/2, align=(:right, :bottom))
    text!(ax, -5.0, 30; text=L"\text{pK}_a = 30",
        fontsize = 6,
          color=:gray, align=(:let, :bottom))

    sc_gen = scatter!(ax, df.homo, df.pKa_DMSO;
        marker=:circle,
        color=MISTStyle.UM_COLORS.blue,
        alpha=0.6,
        label="Generated"
    )

    sc_ref = scatter!(ax, df_ref.homo, df_ref.pKa_DMSO;
        marker=:xcross,
        color=MISTStyle.UM_COLORS.maize,
        alpha=0.8,
        label="Reference"
    )

    Legend(f[2, 1], ax; orientation=:horizontal, tellheight=true, tellwidth=false)

    return f
end

function plot_lithium_air_pka_habstraction(df, df_ref=nothing)
    f = Figure(; size=(2.1inch, 2.1inch), figure_padding=(2, 2, 2, 5))

    df = deepcopy(df)

    # Compute Normalized H-abstraction Rate: rH = exp[-(ln(10)/2)*pKa]
    df.rH = exp.(-(log(10) / 2) .* df.pKa_DMSO)

    ax = Axis(f[1, 1];
        xlabel=L"\text{pK}_a \text{ (DMSO)}",
        ylabel="Normalized H-abstraction Rate",
        yscale=log10
    )

    sc_gen = scatter!(ax, df.pKa_DMSO, df.rH;
        marker=:circle,
        markersize=6,
        color=MISTStyle.UM_COLORS.blue,
        alpha=0.6,
        label="Generated"
    )

    df_ref = deepcopy(df_ref)
    df_ref.rH = exp.(-(log(10) / 2) .* df_ref.pKa_DMSO)

    sc_ref = scatter!(ax, df_ref.pKa_DMSO, df_ref.rH;
        marker=:xcross,
        markersize=8,
        color=MISTStyle.UM_COLORS.maize,
        alpha=0.8,
        label="Reference"
    )

    Legend(f[2, 1], ax; orientation=:horizontal, tellheight=true, tellwidth=false)

    return f
end

function plot_lithium_air_nucleophilic_attack(df, df_ref, equations_module)
    f = Figure(; size=(2.1inch, 2.1inch), figure_padding=(2, 2, 2, 5))

    df = deepcopy(df)

    torch = pyimport("torch")
    pka_tensor = torch.tensor(df.pKa_DMSO)
    beta_tensor = torch.tensor(df.beta)
    etn_tensor = torch.tensor(df.ETN)

    rate_tensor = equations_module.nucleophilic_attack_bound(pka_tensor, beta_tensor, etn_tensor)
    df.rate_nucleophilic = pyconvert(Vector{Float64}, rate_tensor.numpy())

    ax = Axis(f[1, 1];
        xlabel=L"\Delta\;G^{HA}_{sol} \text{ (eV)}",
        ylabel="Rate of Nucleophilic Attack",
        yscale=log10
    )

    vlines!(ax, -0.35; color=:gray, linestyle=:dash, linewidth=1.5)
    hlines!(ax, 1.0; color=:gray, linestyle=:dash, linewidth=1.5)

    text!(ax, -0.35, 10; text=L"\Delta G = -0.35 \text{ eV}",
         fontsize=6,
          color=:gray, rotation=π/2, align=(:center, :bottom))
    text!(ax, -2.0, 1.0; text="Rate = 1", fontsize=6,
          color=:gray, align=(:left, :bottom))

    sc_gen = scatter!(ax, df.delta_G_HA_sol, df.rate_nucleophilic;
        marker=:circle,
        markersize=6,
        color=MISTStyle.UM_COLORS.blue,
        alpha=0.6,
        label="Generated"
    )

    df_ref = deepcopy(df_ref)
    pka_ref_tensor = torch.tensor(df_ref.pKa_DMSO)
    beta_ref_tensor = torch.tensor(df_ref.beta)
    etn_ref_tensor = torch.tensor(df_ref.ETN)

    rate_ref_tensor = equations_module.nucleophilic_attack_bound(pka_ref_tensor, beta_ref_tensor, etn_ref_tensor)
    df_ref.rate_nucleophilic = pyconvert(Vector{Float64}, rate_ref_tensor.numpy())

    sc_ref = scatter!(ax, df_ref.delta_G_HA_sol, df_ref.rate_nucleophilic;
        marker=:xcross,
        markersize=8,
        color=MISTStyle.UM_COLORS.maize,
        alpha=0.8,
        label="Reference"
    )

    Legend(f[2, 1], ax; orientation=:horizontal, tellheight=true, tellwidth=false)

    return f
end

function plot_lithium_air_functional_groups(df, df_ref)
    f = Figure(; size=(4inch, 3inch), figure_padding=(5, 5, 5, 5))

    functional_groups = pyimport("electrolyte_fm.interpretibility.functional_groups")

    gen_counts = Dict{String, Int}()
    for smi in df.smiles
        groups = functional_groups.identify_functional_groups(smi)
        for group in pyconvert(Vector{String}, groups)
            gen_counts[group] = get(gen_counts, group, 0) + 1
        end
    end

    ref_counts = Dict{String, Int}()
    for smi in df_ref.smi
        groups = functional_groups.identify_functional_groups(smi)
        for group in pyconvert(Vector{String}, groups)
            ref_counts[group] = get(ref_counts, group, 0) + 1
        end
    end

    all_groups = unique(vcat(collect(keys(gen_counts)), collect(keys(ref_counts))))
    sort!(all_groups; by = g -> -(get(gen_counts, g, 0) + get(ref_counts, g, 0)))
    gen_values = [max(get(gen_counts, g, 0), 1) for g in all_groups]
    ref_values = [max(get(ref_counts, g, 0), 1) for g in all_groups]

    ax1 = Axis(f[1, 1];
        xlabel="Functional Group",
        ylabel="Count",
        xticks=(1:length(all_groups), titlecase.(all_groups) ),
        xticklabelrotation=π/4,
        yscale=log10,
        xgridvisible=false
    )

    x = 1:length(all_groups)
    barwidth = 0.35
    barplot!(ax1, x .- barwidth/2, gen_values;
        width=barwidth,
        color=MISTStyle.UM_COLORS.blue,
        label="Generated"
    )
    barplot!(ax1, x .+ barwidth/2, ref_values;
        width=barwidth,
        color=MISTStyle.UM_COLORS.maize,
        label="Reference"
    )
    sublabel!(f[1, 1, TopLeft()], "a"; left=5pt)

    gen_fluorinated = count(smi -> occursin('F', smi), df.smiles)
    gen_nonfluorinated = nrow(df) - gen_fluorinated
    ref_fluorinated = count(smi -> occursin('F', smi), df_ref.smi)
    ref_nonfluorinated = nrow(df_ref) - ref_fluorinated

    ax2 = Axis(f[1, 2];
        xlabel="Type",
        ylabel="Count",
        xticks=(1:2, ["Fluorinated", "Non-Fluorinated"]),
        xgridvisible=false
    )

    x2 = 1:2
    gen_fluor_values = [gen_fluorinated, gen_nonfluorinated]
    ref_fluor_values = [ref_fluorinated, ref_nonfluorinated]

    @info "Fluorination counts:" gen_fluorinated gen_nonfluorinated ref_fluorinated ref_nonfluorinated

    barplot!(ax2, x2 .- barwidth/2, gen_fluor_values;
        width=barwidth,
        color=MISTStyle.UM_COLORS.blue,
        label="Generated"
    )
    barplot!(ax2, x2 .+ barwidth/2, ref_fluor_values;
        width=barwidth,
        color=MISTStyle.UM_COLORS.maize,
        label="Reference"
    )
    sublabel!(f[1, 2, TopLeft()], "b"; left=5pt)

    Legend(f[2, 1:2], ax1; orientation=:horizontal, tellheight=true, tellwidth=false)

    return f
end
