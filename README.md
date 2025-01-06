CUDA SGEMM
---

## Intro

This project attempts to build a matrix multiplication kernel from scratch using CUDA, and progressively optimize it to achieve cuBLAS-like performance. The primary goal of this repository is to learn CUDA optimization and gain a deeper understanding of GPU architecture.

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

```sh
./build/kernels.out   # run
ncu --set full --target-processes all ./build/kernels.out   # run with profiling
```

## Performance

### Analysis

The RTX 3090 has a peak performance of ~35.58 TFLOPS for both FP32 and FP16 operations, with a global memory bandwidth of ~936.2 GB/s. To ensure that an SGEMM kernel is compute-bound on the RTX 3090, it requires an arithmetic intensity of at least 38 FLOPs per byte.

Assuming the dimensions of matrices A, B, and C are all 4096,
- Lower bound for IO time = (4096 × 4096 × 3 × 4 bytes) / 936.2 GB/s = 0.2 ms
- Lower bound for compute time = (4096 × 4096 × 4096 × 2 + 4096 × 4096) / 35.58 TFLOPS = 3.9 ms

### GFLOPS

Assume m = n = k = 4096.

| ID   | Kernel | GFLOPS | Time (ms) |
| ---- | ------ | ------ | --------- |
| 1 | naive | 306.568 | 448.314 |
| 2 | global memory coalescing | 2040.133 | 67.368 |
| 3 | shared memory caching | 2869.453 | 47.897 |

### Memory Profile

```sh
ncu \
--metrics lts__t_bytes.sum.per_second,l1tex__t_bytes.sum.per_second \
--set full \
--target-processes all \
./build/kernels.out
```

| ID   | kernel | dram bytes (GB/s) | dram throughput (%) | l1_bytes (TB/s) | lts_bytes (GB/s) |
| ---- | ------ | ----------------- | ------------------- | --------------- | ---------------- |
| 1 | naive | 14.22 | 1.56 | 3.62 | 33.70 |
| 2 | global memory coalescing | 112.59 | 12.38 | 4.35 | 219.17 |
| 3 | shared memory caching | 145.49 | 15.97 | 0.28 | 283.21 |

## Optimization

### 1. naive
- Each thread computes a single element in the C matrix, loading the entire row of the A matrix and the entire column of the B matrix.
- The computation intensity is extremely low, as each pair of loaded variables results in only one Fused Multiply-Add (FMA) operation.
- In the original configuration of a 32 × 32 block, only one block can be placed on a single SM due to the memory requirements. The minimum amount of data to be loaded is approximately 1 MB = (32 × 4096 × 2 + 32 × 32) × 4 bytes. This significantly exceeds the size of the L1 cache. As a result, more data must be loaded from external L2 cache or gmem in practice.

### 2. global memory coalescing
- Threads within the same block are grouped into warps for execution. On modern GPUs, the warp size is typically 32, meaning that every 32 threads are executed together. Notably, thread IDs are assigned based on the x-dimension, meaning threads with consecutive threadIdx.x values are assigned to the same warp.
- If memory accesses by different threads within the same warp are contiguous in gmem, they can be coalesced into a single memory access. Otherwise, multiple cycles are required to complete the accesses individually.
- In Kernel 1, consecutive rows are executed within the same warp, while columns are not. As a result, threads in the same warp access the same column but 32 consecutive rows. However, these 32 rows are not contiguous in gmem (e.g., base, base + 4096, base + 4096 × 2, ...), preventing memory coalescing.
  ```cpp
  // kernel 1
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  ```
- In Kernel 2, the memory access pattern is optimized by making consecutive columns within the same warp. Different threads in the warp share the same row and access adjacent columns, enabling coalesced memory access.
  ```cpp
  // kernel 2
  const int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
  ```
- As a result, the gmem throughput increased from 14.22 GB/s to 112.59 GB/s.

### 3. shared memory caching
- In kernel 1 and kernel 2, the total amount of data moved from gmem is capped at (2 × k × 4 bytes) × m × n, which is significantly higher than the theoretical minimum required for computation. This inefficiency arises because the smem is not fully utilized. By leveraging smem, the number of data accesses to gmem can be significantly reduced.
- To address this, a tiling strategy is adopted. In each iteration:
  - Each block loads a block_size × block_size tile of matrix A and matrix B into smem.
  - The threads within the block use the shared data in smem to compute a portion of the corresponding C matrix.
  - After completing the computation for the current tile, the block shifts by block_size along the k-dimension of matrices A and B, repeating the process for the next tile.
- This process continues until all tiles are processed, completing the computation for matrix C in k / block_size iterations.
- Since all threads within a block reuse the same data loaded into smem, the memory access volume can theoretically be reduced by a factor of 1 / BLOCK_SIZE compared to directly accessing gmem. However, in kernel 2, the L1 cache already achieves a reasonably high hit rate, which limits the additional benefits of shared memory optimization.

## Reference
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Step-by-Step Optimization of CUDA SGEMM](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [cuBLAS Library Doc](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)
- [Kernel Profile Guide](https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf)
