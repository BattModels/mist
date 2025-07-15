struct Asinh
    a::Float64
end
Asinh() = Asinh(1)
(m::Asinh)(x::Real) = m.a * asinh(x / m.a)

Makie.inverse_transform(m::Asinh) = x -> m.a * sinh(x / m.a)
Makie.defined_interval(::Asinh) = Makie.defined_interval(identity)
Makie.defaultlimits(m::Asinh) = (0.0, 10 * m.a)

Makie.inverse_transform(::typeof(asinh)) = sinh
Makie.defined_interval(::typeof(asinh)) = Makie.defined_interval(identity)
Makie.defaultlimits(::typeof(asinh)) = (0.0, 10.0)
