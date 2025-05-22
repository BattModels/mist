@testitem "smirk" begin
    using TokenizerStats: load_tokenizer, pyconvert
    using PythonCall: Py
    tok = load_tokenizer("smirk")
    @test tok isa Py
    @test pyconvert(Vector{String}, tok.tokenize("CO")) == ["C", "O"]
end

@testitem "dataset loader" begin
    using TokenizerStats: DatasetConfig, dataset_split, dataset_name, tokenizer
    using PythonCall: Py, pyconvert

    @testset "tokenizer" begin
        dc = DatasetConfig("qm9", "smirk", "smiles")
        @test dataset_name(dc) == "qm9"
        tok, info = tokenizer(dc)
        @test info.tokenizer_name == "smirk"
        @test info.vocab_size == pyconvert(Int, length(tok)) && info.vocab_size > 0
        @test info.unk_token_id == pyconvert(Union{Int}, tok.unk_token_id) # Smirk has an unk_token_id
    end

    @testset "dataset" begin
        dc = DatasetConfig("qm9", "smirk", "smiles")
        @test dataset_name(dc) == "qm9"
        ds = dataset_split(dc, "train")
        item = first(ds)
        @test item isa Py
        @test haskey(item, "input_ids")
        @test pyconvert(Vector{Int}, item["input_ids"]) isa Vector{Int}
        @test haskey(item, "smi")
        @test pyconvert(String, item["smi"]) isa String
    end
end
