using Libdl
using LinearAlgebra
using SparseArrays

const libQdldlPath = "/home/boom/Robotics/qdldlJulia/qdldl/build/out/libqdldl.so"

struct QDLDLWorkspace{Tf<:AbstractFloat,Ti<:Integer}
  #internal workspace data
  etree::Vector{Ti}
  Lnz::Vector{Ti}
  iwork::Vector{Ti}
  bwork::Vector{Bool}
  fwork::Vector{Tf}

  #L matrix row indices and data
  Ln::Int         #always Int since SparseMatrixCSC does it this way
  Lp::Vector{Ti}
  Li::Vector{Ti}
  Lx::Vector{Tf}

  #D and its inverse
  D::Vector{Tf}
  Dinv::Vector{Tf}

  #number of positive values in D
  positive_inertia::Ref{Ti}

  #The upper triangular matrix factorisation target
  #This is the post ordering PAPt of the original data
  triuA::SparseMatrixCSC{Tf,Ti}

  #mapping from entries in the triu form
  #of the original input to the post ordering
  #triu form used for the factorization
  #this can be used when modifying entries
  #of the data matrix for refactoring
  AtoPAPt::Union{Vector{Ti},Nothing}

  libqdldl::Ptr{Nothing}
end

function QDLDLWorkspace(triuA::SparseMatrixCSC{Tf,Ti},
  AtoPAPt::Union{Vector{Ti},Nothing}
) where {Tf<:AbstractFloat,Ti<:Integer}

  libqdldl = Libdl.dlopen(libQdldlPath)

  @show etree = Vector{Ti}(undef, triuA.n)
  Lnz = Vector{Ti}(undef, triuA.n)
  iwork = Vector{Ti}(undef, triuA.n * 3)
  bwork = Vector{Bool}(undef, triuA.n)
  fwork = Vector{Tf}(undef, triuA.n)

  #compute elimination tree using QDLDL converted code
  QDLDL_etree = Libdl.dlsym(libqdldl, "QDLDL_etree")
  @show Ap = triuA.colptr .- 1
  @show Ai = triuA.rowval .- 1
  sumLnz = ccall(QDLDL_etree, Clonglong, (Clonglong, Ref{Clonglong}, Ref{Clonglong}, Ref{Clonglong}, Ref{Clonglong}, Ref{Clonglong}), triuA.n, Ap, Ai, iwork, Lnz, etree)

  @show sumLnz
  @show etree
  if (sumLnz < 0)
    error("Input matrix is not upper triangular or has an empty column")
  end

  #allocate space for the L matrix row indices and data
  Ln = triuA.n
  Lp = Vector{Ti}(undef, triuA.n + 1)
  Li = Vector{Ti}(undef, sumLnz)
  Lx = Vector{Tf}(undef, sumLnz)

  #allocate for D and D inverse
  D = Vector{Tf}(undef, triuA.n)
  Dinv = Vector{Tf}(undef, triuA.n)

  #allocate for positive inertia count.  -1 to
  #start since we haven't counted anything yet
  positive_inertia = Ref{Ti}(-1)

  QDLDLWorkspace(etree, Lnz, iwork, bwork, fwork, Ln, Lp, Li, Lx, D, Dinv, positive_inertia, triuA, AtoPAPt, libqdldl)

end

A = triu(sprand(3, 3, 0.5) + sparse(I * 1.0, 3, 3))
@show A.n, A.colptr, A.rowval
workspace = QDLDLWorkspace(A, nothing)

@show workspace.etree