#include <cuda_runtime.h>
#include "sgemm.cuh"


template <const int BLOCKSIZE>
__global__ void sgemm_global_mem_coalescing(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    const int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    if ((row < m) && (col < n)) {
        float value = 0.0;
        for (int i = 0; i < k; i++) {
            value += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = alpha * value + beta * c[row * n + col];
    }
}

// warp size = 32
template __global__ void sgemm_global_mem_coalescing<32>(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);
