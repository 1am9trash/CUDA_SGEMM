CUDA SGEMM
---

## Intro

This project attempts to build a matrix multiplication kernel from scratch using CUDA, and progressively optimize it to achieve cuBLAS-like performance. The primary goal of this repository is to learn CUDA optimization and gain a deeper understanding of GPU architecture, including coalescing global memory accesses, shared memory caching, and occupancy optimizations.

The focus is initially on single-precision GEMM (SGEMM), with plans to extend the work to sparse matrix multiplication in the future.

## SGEMM

SGEMM stands for Single-precision General Matrix Multiplication. It computes the matrix product of two matrices, followed by scaling and accumulation according to the following formula.

```
A: m x k  
B: k x n  
C: m x n  

C = alpha * (A @ B) + beta * C
```

## Run

Use the Makefile to build the executable.

```sh
make        # Compile the source code and generate the executable at build/kernels.out
make clean  # Remove all files in the build folder
```

## Reference
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Step-by-Step Optimization of CUDA SGEMM](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [cuBLAS Library Doc](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)
