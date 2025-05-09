function df_ngram_stats_v_fm_perf(tok_info, model_loss, info_loss, df_f; reference="character")
    info_loss = subset(info_loss,
        :ngram => ByRow(==(5)),
        :ref_tokenizer => ByRow(==(reference)),
    )
    model_loss = subset(model_loss,
        :dataset => ByRow(!=("realspace")),
        :split => ByRow(==("val")),
        :ngram => ByRow(==(5)),
    )
    df = leftjoin(
        select(info_loss, :tokenizer, :dataset, :info_loss_moments),
        select(model_loss, :tokenizer, :dataset, :finetuned, :loss_per_token_moments);
        on=[:tokenizer, :dataset],
    )
    disallowmissing!(df)
    transform!(df,
        :tokenizer => ByRow(x -> tok_info[x]["tokenizer_class"]) => :tokenizer_class,
        :tokenizer => ByRow(x -> tok_info[x]["encoding"]) => :encoding,
        :info_loss_moments => ByRow(mean) => :info_loss_avg,
        :info_loss_moments => ByRow(std) => :info_loss_std,
        :loss_per_token_moments => ByRow(mean) => :ng_loss_avg,
        :loss_per_token_moments => ByRow(std) => :ng_loss_std,
    )
    select!(df, Not([:info_loss_moments, :loss_per_token_moments]))
    df_f = select(df_f, :tokenizer, :dataset, :encoding, :metric, :mean => :fm_loss_avg, :std => :fm_loss_std)
    return leftjoin!(df, df_f; on=[:tokenizer, :dataset, :encoding])
end

function ngram_prognostic_fits(df_prog)
    df_prog = select(df_prog, [:finetuned, :dataset, :encoding, :ng_loss_avg, :info_loss_avg, :fm_loss_avg, :fm_loss_std])
    dropmissing!(df_prog)
    df = combine(groupby(df_prog, [:dataset, :finetuned])) do gdf
        wts = aweights(inv.(gdf.fm_loss_std .^ 2))
        loss_only = lm(@formula(fm_loss_avg ~ 1 + ng_loss_avg), gdf; wts)
        loss_and_info = lm(@formula(fm_loss_avg ~ 1 + ng_loss_avg + info_loss_avg), gdf; wts)
        return (; loss_only, loss_and_info)
    end
    df_ft = combine(groupby(df_prog, :dataset)) do gdf
        gdf = unstack(gdf, :finetuned, :ng_loss_avg, renamecols=x -> x ? :ng_ft_loss_avg : :ng_loss_avg)
        wts = aweights(inv.(gdf.fm_loss_std .^ 2))
        loss_and_info_and_ft = lm(@formula(fm_loss_avg ~ 1 + ng_loss_avg + ng_ft_loss_avg + info_loss_avg), gdf; wts)
        loss_and_ft = lm(@formula(fm_loss_avg ~ 1 + ng_loss_avg + ng_ft_loss_avg), gdf; wts)
        return (; loss_and_info_and_ft, loss_and_ft)
    end
    leftjoin!(df, df_ft; on=:dataset)
    transform!(df,
        :loss_only => ByRow(r2),
        :loss_and_info => ByRow(r2),
        :loss_and_info_and_ft => ByRow(r2),
        :loss_and_ft => ByRow(r2),
        # [:loss_only, :loss_and_info_and_ft] => ByRow(ftests) => :ftests,
        :loss_and_info_and_ft => ByRow(m -> SpearmanTTest(m).rho) => :all_rho,
        :loss_and_info_and_ft => ByRow(pvalue∘SpearmanTTest) => :all_rho_p,
        :loss_only => ByRow(nobs_nonwts) => :nobs,
        :loss_only => ByRow(StatsBase.variation ∘ response) => :response_cv,
    )
    return df
end

function ftests(models...)
    ftest(getfield.(models, :model)...)
end

struct SpearmanTTest
    rho::Float64
    t::Float64
    nobs::Int
end

SpearmanTTest(model::StatsBase.RegressionModel) = SpearmanTTest(response(model), predict(model))
function SpearmanTTest(x::AbstractVector, y::AbstractVector)
    rho = corspearman(x, y)
    nobs = length(x)
    t = rho * sqrt((nobs - 2) / (1 - rho ^ 2))
    return SpearmanTTest(rho, t, nobs)
end

HypothesisTests.pvalue(stt::SpearmanTTest) = pvalue(TDist(stt.nobs-2), stt.t)

function Base.show(io::IO, mime::MIME"text/plain", stt::SpearmanTTest)
    println(io, "ρ:       $(round(stt.rho; sigdigits=3))")
    println(io, "n:       $(stt.nobs)")
    println(io, "t:       $(round(stt.t; sigdigits=3))")
    println(io, "p-value: $(round(pvalue(stt); sigdigits=3))")
end

