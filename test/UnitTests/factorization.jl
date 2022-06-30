using QDLDL_CCALL
using LinearAlgebra: I, inv, transpose
using SparseArrays: permute

@testset "factorization AMD perm" begin

  m = 20

  A = random_psd(m)
  F = qdldl(A)

  PAPt = (F.L + I) * inv(F.Dinv) * transpose(F.L + I)

  @test PAPt ≈ permute(A, F.perm, F.perm) atol = 1e-14

end

@testset "factorization no perm" begin

  m = 20

  A = random_psd(m)
  F = qdldl(A, perm=nothing)

  @test (F.L + I) * inv(F.Dinv) * transpose(F.L + I) ≈ A atol = 1e-14

end