sci_notation(x::Vector; kwargs...) = map(sci_notation(; kwargs...), x)
sci_notation(; sigdigits=2) = x -> sci_notation(x; sigdigits)

function sci_notation(x::Real; sigdigits=2)
    if x != zero(x)
        lnx = log10(abs(x))
        exponent = floor(Int, lnx)
    else
        exponent = 0
    end
    mantissa = x / float(10)^exponent
    sign = x < 0 ? L"\text{-}" : L"\text{ }"
    mantissa = round(abs(mantissa), sigdigits=sigdigits)
    return L"%$sign%$mantissa\times10^{%$exponent}"
end
