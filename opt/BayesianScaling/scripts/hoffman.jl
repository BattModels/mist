using BayesianScaling: BayesianScaling, HoffmanScaling
using DataFrames
using CSV: CSV
using Downloads: Downloads
using Enzyme: Enzyme

url = "https://raw.githubusercontent.com/scifm/summer-school-2024/main/data/scaling_law.csv"
ds_file = Downloads.download(url)
df = DataFrame(CSV.File(ds_file))
select!(df, :model_size, :total_tokens => :data_size, :loss)
display(df)

model = BayesianScaling.init_model(HoffmanScaling(), df)
display(model)

y = BayesianScaling.sample_chains(model; nchains=4, draws=1000)
display(y)

