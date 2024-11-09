#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel_runner.cuh"
#include "kernels/sgemm.cuh"


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

void run_1_sgemm_naive(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    dim3 grid_dim(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    dim3 block_dim(32, 32);
    sgemm_naive<<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_2_sgemm_global_mem_coalescing(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BLOCKSIZE = 32;
    dim3 grid_dim(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    dim3 block_dim(BLOCKSIZE * BLOCKSIZE);
    sgemm_global_mem_coalescing<BLOCKSIZE><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_sgemm(
    int kernel_id,
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    if (kernel_id == 1)
        run_1_sgemm_naive(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 2)
        run_2_sgemm_global_mem_coalescing(m, n, k, alpha, a, b, beta, c);
}
