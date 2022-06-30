
using Test, LinearAlgebra, SparseArrays, Random
using QDLDL_CCALL
rng = Random.MersenneTwister(0706)

@testset "linear solves" begin

  m = 20
  n = 30

  A = random_psd(n)
  B = sprandn(m, n, 0.2)
  C = -random_psd(m)
  M = [A B'; B C]

  b = randn(m + n)
  F = qdldl(M)

  #basic solve
  @test norm(F \ b - M \ b, Inf) <= 1e-10

  #solve function
  @test norm(solve(F, b) - M \ b, Inf) <= 1e-10

  #solve in place
  x = copy(b)
  solve!(F, x)
  @test norm(x - M \ b, Inf) <= 1e-10

  #scalar system
  A = sprandn(1, 1, 1.0)
  b = randn(1)
  @test norm(qdldl(A) \ b - A \ b, Inf) <= 1e-10


end

nothing
