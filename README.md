# QDLDL_CCALL.jl

## Get started

1. Clone the repo
```
git clone --recurse-submodules git@github.com:Huoleit/QDLDL_CCALL.jl.git
```

2. Compile C shared library
```
sh ./compile_binary.sh  
```

3. Run all Julia test to make sure everything works as expected

## Usage 

qdldl(A) uses the default AMD ordering

qdldl(A,perm=p) uses a caller specified ordering

qdldl(A,perm = nothing) factors without reordering

**See unit tests for other usages.**
