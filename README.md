# QDLDL_CCALL.jl

## Get started

1. Compile C shared library first
```
sh ./compile_binary.sh  
```

2. Run all Julia test to make sure everything works as expected

## Usage 

qdldl(A) uses the default AMD ordering

qdldl(A,perm=p) uses a caller specified ordering

qdldl(A,perm = nothing) factors without reordering
