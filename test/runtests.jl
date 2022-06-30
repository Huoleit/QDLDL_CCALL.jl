include("../src/QDLDL_CCALL.jl")

import .QDLDL_CCALL
using SparseArrays
using LinearAlgebra
using Test

@testset "QDLDL" begin
  @testset "No permutation" begin
    An = 15
    mat = sprand(An, An, 0.5) + sparse(I * 1.0, An, An)
    A = transpose(mat) * mat

    fac = QDLDL_CCALL.qdldl(A, perm=nothing)
    a = (fac.L + I) * inv(fac.Dinv) * transpose(fac.L + I)
    @test A ≈ a atol = 1e-14
  end

  @testset "AMD permutation" begin
    An = 15
    mat = sprand(An, An, 0.5) + sparse(I * 1.0, An, An)
    A = transpose(mat) * mat

    fac = QDLDL_CCALL.qdldl(A)
    a = (fac.L + I) * inv(fac.Dinv) * transpose(fac.L + I)
    @test permute(A, fac.perm, fac.perm) ≈ a atol = 1e-14
  end
end

