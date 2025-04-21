#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script
using MISTStyle: MISTStyle

include("token_embeddings.jl")
include("embedding_figure.jl")

models = [
    "Pretrained" => "../../models/mist-ti624ev1-moleculenet/pretrained",
    "tmQM" => "../../models/mist-ti624ev1-moleculenet/tmqm",
    "QM9" => "../../models/mist-ti624ev1-moleculenet/qm9",
    "FreeSolv" => "../../models/mist-ti624ev1-moleculenet/freesolv",
    "QM8" => "../../models/mist-ti624ev1-moleculenet/qm8",
    "Lipo" => "../../models/mist-ti624ev1-moleculenet/lipo",
    "ToxCast" => "../../models/mist-ti624ev1-moleculenet/toxcast",
    "BACE" => "../../models/mist-ti624ev1-moleculenet/bace",
    "ESOL" => "../../models/mist-ti624ev1-moleculenet/esol",
    "HIV" => "../../models/mist-ti624ev1-moleculenet/hiv",
    "Tox21" => "../../models/mist-ti624ev1-moleculenet/tox21",
    "SIDER" => "../../models/mist-ti624ev1-moleculenet/sider",
    "MUV" => "../../models/mist-ti624ev1-moleculenet/muv",
    "BBBP" => "../../models/mist-ti624ev1-moleculenet/bbbp",
]
with_theme(MISTStyle.theme()) do
    figure_token_embeddings(models[1:8]; emb_models=4)
end |> MISTStyle.savefig("token_embeddings_updates")
with_theme(MISTStyle.theme()) do
    figure_token_embeddings(models; last_token=Inf, fig_size=(7inch, 4.5inch), min_update=1e-6)
end |> MISTStyle.savefig("token_embeddings_updates_si")


with_theme(MISTStyle.theme()) do
    figure_embedding()
end |> MISTStyle.savefig("interp_embeddings")
