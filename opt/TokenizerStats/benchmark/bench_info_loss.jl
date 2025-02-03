using BenchmarkTools
using TokenizerStats
using TokenizerStats: NGramModel, SlidingWindow
using OnlineStats

const suite = BenchmarkGroup()

# Build random n-gram counts
function randngram(N=3; vocab_size=9, corpus=64)
    stats = map(n -> CountMap(NTuple{n,Int}), 1:N)
    corpus = rand(range(0; length=vocab_size), 8, corpus)
    for i in axes(corpus, 2)
        for (n, s) in enumerate(stats)
            fit!(s, SlidingWindow(corpus[:, i], n))
        end
    end
    return map(value, stats)
end


mask = -100
code = rand((1, 2, 3, 4, 5), 70)
mask = Bool.(rand(length(code)) .>= 0.7)
vocab_size = 2000
m = NGramModel(randngram(5; vocab_size), vocab_size)

for N in 1:length(m)
    suite["info_loss_$N"] = @benchmarkable TokenizerStats.information_loss($m, $code, $mask; N=$N)
end
