"""
    read_xyz(path::AbstractString) -> DataFrame
    read_xyz(io::IO) -> DataFrame

Parse a plain XYZ file into a DataFrame with columns:
:element::String, :x::Float64, :y::Float64, :z::Float64.

- Skips the first (atom count) and second (comment) lines.
- Ignores blank lines and lines starting with `#`.
- If extra columns exist on coordinate lines, they are ignored.
- Warns if the header atom count doesn't match parsed rows.
"""
read_xyz(path::AbstractString) = open(read_xyz, path, "r")

function read_xyz(io::IO)
    # 1) Atom count line (optional validation)
    n_header = tryparse(Int, strip(readline(io)))

    # 2) Comment line (drop it if present)
    if !eof(io)
        _ = readline(io)  # discard the comment line
    end

    rows = Vector{NamedTuple{(:element, :x, :y, :z),
                             Tuple{String, Float64, Float64, Float64}}}()

    for line in eachline(io)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue  # allow inline comments after header

        toks = split(s)
        length(toks) < 4 && continue     # skip malformed lines

        push!(rows, (element = toks[1],
                     x = parse(Float64, toks[2]),
                     y = parse(Float64, toks[3]),
                     z = parse(Float64, toks[4])))
    end

    df = DataFrame(rows)

    if n_header !== nothing && n_header != nrow(df)
        @warn "Atom count in header ($n_header) does not match rows parsed ($(nrow(df)))."
    end

    return df
end


function split_xyz(input_path::AbstractString; outdir::AbstractString=".", overwrite::Bool=false)
    isdir(outdir) || mkpath(outdir)

    lines = readlines(input_path)
    i = 1
    written = String[]
    nlines = length(lines)

    csd_re = r"CSD_code\s*=\s*([^\s|]+)"  # capture CODE until whitespace or '|'

    while i <= nlines
        n_atoms = tryparse(Int, strip(lines[i]))
        if n_atoms === nothing
            i += 1
            continue
        end

        if i + 1 + n_atoms > nlines
            @warn "Truncated block at line $i: expected $(n_atoms) atoms but file ended early."
            break
        end

        comment = lines[i + 1]
        atom_block = lines[(i + 2):(i + 1 + n_atoms)]

        m = match(csd_re, comment)
        csd_code = m === nothing ? "block_$(length(written) + 1)" : m.captures[1]

        base = replace(csd_code, r"[^A-Za-z0-9._+-]" => "_")
        target = joinpath(outdir, base * ".xyz")

        if !overwrite && isfile(target)
            k = 1
            while isfile(joinpath(outdir, base * "-" * string(k) * ".xyz"))
                k += 1
            end
            target = joinpath(outdir, base * "-" * string(k) * ".xyz")
        end

        open(target, "w") do io
            println(io, n_atoms)
            println(io, comment)
            for l in atom_block
                println(io, l)
            end
        end
        push!(written, target)

        i += 2 + n_atoms
    end

    return written
end
