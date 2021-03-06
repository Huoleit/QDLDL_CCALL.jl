using Random, Test
rng = Random.MersenneTwister(12345)

function random_psd(n)

  A = sprandn(rng, n, n, 0.2)
  A = A + A'
  A = A + Diagonal(((sum(abs.(A), dims=1)[:]))) #make diagonally dominant

end


@testset "All Unit Tests" begin

  include("./UnitTests/basic.jl")
  include("./UnitTests/factorization.jl")
  include("./UnitTests/non-quasidef.jl")
  include("./UnitTests/inertia.jl")

end
