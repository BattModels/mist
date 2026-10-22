using Test
include(joinpath(@__DIR__, "..", "plots.jl"))

@testset "Original pretraining figure inputs" begin
    mktempdir() do dir
        source = joinpath(dir, "gas.csv")
        gas = DataFrame([(run_id="$(gpus)-$(gas)", gpus, gas, microbatch_size=512,
                          samples_per_second=1000.0*gpus) for gas in [1,64] for gpus in [64,256,512]])
        CSV.write(source, gas)
        @test nrow(gas_observations(source)) == 6
        CSV.write(source, gas[gas.gpus .<= 64, :])
        @test_throws ErrorException gas_observations(source)
        CSV.write(source, vcat(gas, gas[1:1, :]))
        @test_throws ErrorException gas_observations(source)

        ablation = DataFrame([(run_id="$dataset-$tokenizer-$position", dataset, tokenizer,
                               position_embedding=position,
                               metric=dataset in ["esol", "freesolv", "qm9"] ? "r2" : "auroc", value=0.8)
                              for dataset in ABLATION_TASKS for (tokenizer, position) in ABLATION_VARIANTS])
        source = joinpath(dir, "ablation.csv")
        CSV.write(source, ablation)
        @test nrow(ablation_observations(source)) == 32
        ablation.metric[1] = "accuracy"
        CSV.write(source, ablation)
        @test_throws ErrorException ablation_observations(source)

        losses = DataFrame([(run_id=condition, condition, step, train_loss=1.0, spike_step=2)
                            for condition in ["Spiking", "Corrected"] for step in 0:4])
        eigenvalues = DataFrame([(run_id=condition, condition, step, matrix_group=group,
                                 layer, matrix=group, lambda_max=0.1*layer)
                                for condition in ["Spiking", "Corrected"] for group in ["Feedforward", "KQV"]
                                for step in 0:4 for layer in 1:3])
        loss_path, eigenvalue_path = joinpath(dir, "loss.csv"), joinpath(dir, "lambda.csv")
        CSV.write(loss_path, losses)
        CSV.write(eigenvalue_path, eigenvalues)
        @test nrow(last(instability_observations(loss_path, eigenvalue_path))) == 60
        eigenvalues.run_id[1] = "unrelated-run"
        CSV.write(eigenvalue_path, eigenvalues)
        @test_throws ErrorException instability_observations(loss_path, eigenvalue_path)
        @test_throws ErrorException gas_observations(joinpath(dir, "absent.csv"))
    end
end

@testset "Released transfer observations" begin
    source = joinpath(@__DIR__, "..", "..", "..", "data", "releases", "zenodo-21302713",
                      "extracted", "scaling", "mist_pretraining_advantage.csv")
    if isfile(source)
        data = transfer_observations(source)
        @test nrow(data) == 78
        @test Set(data.model_size) == Set(["28M", "1.8B"])
        @test extrema(data.samples) == (6928, 69328)
        @test nrow(subset(data, :frozen => ByRow(!))) == 38
    else
        @test_skip "Released transfer CSV is not staged"
    end
end
