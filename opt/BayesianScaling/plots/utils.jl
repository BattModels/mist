using Printf

function sn(x::Real; sigdigits::Int=3, cmd::AbstractString="\\sn")
    @assert sigdigits ≥ 1 "sigdigits must be ≥ 1"

    # Special values
    if isnan(x)
        return "NaN"
    elseif isinf(x)
        return x > 0 ? "Inf" : "-Inf"
    elseif x == 0
        return "$cmd{0}{0}"
    end

    # %e precision counts digits after the decimal → sigdigits - 1
    s = @sprintf("%.*e", sigdigits - 1, x)

    # Split mantissa/exponent
    m, e = split(s, 'e'; limit=2)

    # Clean mantissa: drop trailing zeros and an orphan decimal point
    m = replace(m, r"\.?0+$" => "")

    # Clean exponent:
    #  - strip '+' if present
    #  - strip leading zeros after optional sign
    e = replace(e, r"^\+?" => "")       # remove '+'
    e = replace(e, r"^(-?)0+" => s"\1") # remove leading zeros, keep '-'

    return "$cmd{$m}{$e}"
end
