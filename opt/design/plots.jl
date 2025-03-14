using Makie
using DataFrames
using CSV

function plot_sat_fats(file)
    f = Figure(size=96 .* (4.5, 6))
    df = DataFrame(CSV.File(file))
    df.n_carbon = map(x -> count(==('C'), x), df.smi)

    ax = Axis(f[1, 1], ylabel="Diploe Moment [D]", xlabel="Number of Carbons")
    scatter!(ax, df.n_carbon, df.mu)

    ax = Axis(f[2, 1], ylabel="Energy [Ha]", xlabel="Number of Carbons")
    scatter!(ax, df.n_carbon, df.u0; marker=:+, label="Internal 0K")
    scatter!(ax, df.n_carbon, df.u298; marker=:x, label="Internal at 298.15K")
    scatter!(ax, df.n_carbon, df.h298; marker=:circle, label="Enthalpy at 298.15K")
    scatter!(ax, df.n_carbon, df.g298; marker=:diamond, label="Gibbs at 298.15K")
    axislegend(ax; position=:rb, nbanks=2)

    ax = Axis(f[3, 1], ylabel="HOMO-LUMO Gap", xlabel="Number of Carbons")
    scatter!(ax, df.n_carbon, df.gap; marker=:x, label="Direct")
    scatter!(ax, df.n_carbon, df.lumo - df.homo; marker=:+, label="Calculated")
    axislegend(ax; position=:rb)
    return f
end
