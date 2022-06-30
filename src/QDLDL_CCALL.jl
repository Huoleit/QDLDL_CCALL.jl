module QDLDL_CCALL

export qdldl, \, solve, solve!, refactor!, positive_inertia

using AMD, SparseArrays
using LinearAlgebra: istriu, triu, Diagonal

const libQdldlPath = joinpath(dirname(@__FILE__), "qdldl/build/out/libqdldl.so")

struct QDLDLWorkspace{Tf<:AbstractFloat,Ti<:Integer}
  #internal workspace data
  etree::Vector{Ti}
  Lnz::Vector{Ti}
  iwork::Vector{Ti}
  bwork::Vector{UInt8}
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
end

function QDLDLWorkspace(triuA::SparseMatrixCSC{Tf,Ti},
  AtoPAPt::Union{Vector{Ti},Nothing}
) where {Tf<:AbstractFloat,Ti<:Integer}

  etree = Vector{Ti}(undef, triuA.n)
  Lnz = Vector{Ti}(undef, triuA.n)
  iwork = Vector{Ti}(undef, triuA.n * 3)
  bwork = Vector{UInt8}(undef, triuA.n)
  fwork = Vector{Tf}(undef, triuA.n)

  #compute elimination tree using QDLDL code
  Ap::Vector{Int64} = triuA.colptr .- 1
  Ai::Vector{Int64} = triuA.rowval .- 1
  sumLnz = ccall(("QDLDL_etree", libQdldlPath), Int64, (Int64, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}), triuA.n, Ap, Ai, iwork, Lnz, etree)

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

  QDLDLWorkspace(etree, Lnz, iwork, bwork, fwork, Ln, Lp, Li, Lx, D, Dinv, positive_inertia, triuA, AtoPAPt)
end

struct QDLDLFactorisation{Tf<:AbstractFloat,Ti<:Integer}

  #permutation vector (nothing if no permutation)
  perm::Union{Nothing,Vector{Ti}}
  #inverse permutation (nothing if no permutation)
  iperm::Union{Nothing,Vector{Ti}}
  #lower triangular factor
  L::SparseMatrixCSC{Tf,Ti}
  #Inverse of D matrix in ldl
  Dinv::Diagonal{Tf,Vector{Tf}}
  #workspace data
  workspace::QDLDLWorkspace{Tf,Ti}
end

# Usage :
# qdldl(A) uses the default AMD ordering
# qdldl(A,perm=p) uses a caller specified ordering
# qdldl(A,perm = nothing) factors without reordering
#
# qdldl(A,logical=true) produces a logical factorisation only
#
# qdldl(A,signs = s, thresh_eps = ϵ, thresh_delta = δ) produces
# a factorization with dynamic regularization based on the vector
# of signs in s and using regularization parameters (ϵ,δ).  The
# scalars (ϵ,δ) = (1e-12,1e-7) by default.   By default s = nothing,
# and no regularization is performed.

function qdldl(A::SparseMatrixCSC{Tf,Ti};
  perm::Union{Array{Ti},Nothing}=amd(A)
) where {Tf<:AbstractFloat,Ti<:Integer}

  #store the inverse permutation to enable matrix updates
  iperm = perm == nothing ? nothing : invperm(perm)

  if (!istriu(A))
    A = triu(A)
  end

  #permute using symperm, producing a triu matrix to factor
  if perm != nothing
    A, AtoPAPt = permute_symmetric(A, iperm)  #returns an upper triangular matrix
  else
    AtoPAPt = nothing
  end

  #allocate workspace
  workspace = QDLDLWorkspace(A, AtoPAPt)

  #factor the matrix
  factor!(workspace)

  #make user-friendly factors
  L = SparseMatrixCSC(workspace.Ln,
    workspace.Ln,
    workspace.Lp,
    workspace.Li,
    workspace.Lx)
  Dinv = Diagonal(workspace.Dinv)

  return QDLDLFactorisation(perm, iperm, L, Dinv, workspace)

end

function positive_inertia(F::QDLDLFactorisation)
  F.workspace.positive_inertia[]
end

function Base.:\(F::QDLDLFactorisation, b)
  return solve(F, b)
end


function refactor!(F::QDLDLFactorisation)

  #It never makes sense to call refactor for a logical
  #factorization since it will always be the same.  Calling
  #this function implies that we want a numerical factorization

  factor!(F.workspace)
end


function factor!(workspace::QDLDLWorkspace{Tf,Ti}) where {Tf<:AbstractFloat,Ti<:Integer}
  #factor using QDLDL converted code
  A = workspace.triuA
  Ap = A.colptr .- 1
  Ai = A.rowval .- 1

  posDCount = ccall(("QDLDL_factor", libQdldlPath),
    Int64,
    (
      Int64,        #const QDLDL_int    n,
      Ref{Int64},   #const QDLDL_int*   Ap,
      Ref{Int64},   #const QDLDL_int*   Ai,
      Ref{Float64}, #const QDLDL_float* Ax,
      Ref{Int64},   #QDLDL_int*   Lp,
      Ref{Int64},   #QDLDL_int*   Li,
      Ref{Float64}, #QDLDL_float* Lx,
      Ref{Float64}, #QDLDL_float* D,
      Ref{Float64}, #QDLDL_float* Dinv,
      Ref{Int64},   #const QDLDL_int* Lnz,
      Ref{Int64},   #const QDLDL_int* etree,
      Ref{UInt8},   #QDLDL_bool* bwork,
      Ref{Int64},   #QDLDL_int* iwork,
      Ref{Float64}  #QDLDL_float* fwork
    ),
    A.n,
    Ap,
    Ai,
    A.nzval,
    workspace.Lp,
    workspace.Li,
    workspace.Lx,
    workspace.D,
    workspace.Dinv,
    workspace.Lnz,
    workspace.etree,
    workspace.bwork,
    workspace.iwork,
    workspace.fwork
  )

  workspace.Lp .+= 1
  workspace.Li .+= 1

  if (posDCount < 0)
    error("Zero entry in D (matrix is not quasidefinite)")
  end

  workspace.positive_inertia[] = posDCount

  return nothing

end


# Solves Ax = b using LDL factors for A.
# Returns x, preserving b
function solve(F::QDLDLFactorisation, b)
  x = copy(b)
  solve!(F, x)
  return x
end

# Solves Ax = b using LDL factors for A.
# Solves in place (x replaces b)
function solve!(F::QDLDLFactorisation, b)

  #permute b
  tmp = F.perm == nothing ? b : permute!(F.workspace.fwork, b, F.perm)

  QDLDL_solve!(F.workspace.Ln,
    F.workspace.Lp,
    F.workspace.Li,
    F.workspace.Lx,
    F.workspace.Dinv,
    tmp)

  #inverse permutation
  b = F.perm == nothing ? tmp : ipermute!(b, F.workspace.fwork, F.perm)

  return nothing
end

# Solves (L+I)x = b, with x replacing b
function QDLDL_Lsolve!(n, Lp, Li, Lx, x)

  @inbounds for i = 1:n
    @inbounds for j = Lp[i]:(Lp[i+1]-1)
      x[Li[j]] -= Lx[j] * x[i]
    end
  end
  return nothing
end


# Solves (L+I)'x = b, with x replacing b
function QDLDL_Ltsolve!(n, Lp, Li, Lx, x)

  @inbounds for i = n:-1:1
    @inbounds for j = Lp[i]:(Lp[i+1]-1)
      x[i] -= Lx[j] * x[Li[j]]
    end
  end
  return nothing
end

# Solves Ax = b where A has given LDL factors,
# with x replacing b
function QDLDL_solve!(n, Lp, Li, Lx, Dinv, b)

  QDLDL_Lsolve!(n, Lp, Li, Lx, b)
  b .*= Dinv
  QDLDL_Ltsolve!(n, Lp, Li, Lx, b)

end



# internal permutation and inverse permutation
# functions that require no memory allocations
function permute!(x, b, p)
  @inbounds for j = 1:length(x)
    x[j] = b[p[j]]
  end
  return x
end

function ipermute!(x, b, p)
  @inbounds for j = 1:length(x)
    x[p[j]] = b[j]
  end
  return x
end


"Given a sparse symmetric matrix `A` (with only upper triangular entries), return permuted sparse symmetric matrix `P` (only upper triangular) given the inverse permutation vector `iperm`."
function permute_symmetric(
  A::SparseMatrixCSC{Tf,Ti},
  iperm::AbstractVector{Ti},
  Pr::AbstractVector{Ti}=zeros(Ti, nnz(A)),
  Pc::AbstractVector{Ti}=zeros(Ti, size(A, 1) + 1),
  Pv::AbstractVector{Tf}=zeros(Tf, nnz(A))
) where {Tf<:AbstractFloat,Ti<:Integer}

  # perform a number of argument checks
  m, n = size(A)
  m != n && throw(DimensionMismatch("Matrix A must be sparse and square"))

  isperm(iperm) || throw(ArgumentError("pinv must be a permutation"))

  if n != length(iperm)
    throw(DimensionMismatch("Dimensions of sparse matrix A must equal the length of iperm, $((m,n)) != $(iperm)"))
  end

  #we will record a mapping of entries from A to PAPt
  AtoPAPt = zeros(Ti, length(Pv))

  P = _permute_symmetric(A, AtoPAPt, iperm, Pr, Pc, Pv)
  return P, AtoPAPt
end

# the main function without extra argument checks
# following the book: Timothy Davis - Direct Methods for Sparse Linear Systems
function _permute_symmetric(
  A::SparseMatrixCSC{Tf,Ti},
  AtoPAPt::AbstractVector{Ti},
  iperm::AbstractVector{Ti},
  Pr::AbstractVector{Ti},
  Pc::AbstractVector{Ti},
  Pv::AbstractVector{Tf}
) where {Tf<:AbstractFloat,Ti<:Integer}

  # 1. count number of entries that each column of P will have
  n = size(A, 2)
  num_entries = zeros(Ti, n)
  Ar = A.rowval
  Ac = A.colptr
  Av = A.nzval

  # count the number of upper-triangle entries in columns of P, keeping in mind the row permutation
  for colA = 1:n
    colP = iperm[colA]
    # loop over entries of A in column A...
    for row_idx = Ac[colA]:Ac[colA+1]-1
      rowA = Ar[row_idx]
      rowP = iperm[rowA]
      # ...and check if entry is upper triangular
      if rowA <= colA
        # determine to which column the entry belongs after permutation
        col_idx = max(rowP, colP)
        num_entries[col_idx] += one(Ti)
      end
    end
  end
  # 2. calculate permuted Pc = P.colptr from number of entries
  Pc[1] = one(Ti)
  @inbounds for k = 1:n
    Pc[k+1] = Pc[k] + num_entries[k]

    # reuse this vector memory to keep track of free entries in rowval
    num_entries[k] = Pc[k]
  end
  # use alias
  row_starts = num_entries

  # 3. permute the row entries and position of corresponding nzval
  for colA = 1:n
    colP = iperm[colA]
    # loop over rows of A and determine where each row entry of A should be stored
    for rowA_idx = Ac[colA]:Ac[colA+1]-1
      rowA = Ar[rowA_idx]
      # check if upper triangular
      if rowA <= colA
        rowP = iperm[rowA]
        # determine column to store the entry
        col_idx = max(colP, rowP)

        # find next free location in rowval (this results in unordered columns in the rowval)
        rowP_idx = row_starts[col_idx]

        # store rowval and nzval
        Pr[rowP_idx] = min(colP, rowP)
        Pv[rowP_idx] = Av[rowA_idx]

        #record this into the mapping vector
        AtoPAPt[rowA_idx] = rowP_idx

        # increment next free location
        row_starts[col_idx] += 1
      end
    end
  end
  nz_new = Pc[end] - 1
  P = SparseMatrixCSC{Tf,Ti}(n, n, Pc, Pr[1:nz_new], Pv[1:nz_new])

  return P
end

end # end module