#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

void run_3_sgemm_shared_mem_caching(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BLOCKSIZE = 32;
    dim3 grid_dim(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    dim3 block_dim(BLOCKSIZE * BLOCKSIZE);
    sgemm_shared_mem_caching<BLOCKSIZE><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_4_sgemm_thread_tiling_1d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BM = 64, BK = 8, BN = 64, TM = 8;
    dim3 grid_dim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_dim(BM * BN / TM);
    sgemm_thread_tiling_1d<BM, BK, BN, TM><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_5_sgemm_thread_tiling_2d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
    dim3 grid_dim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_dim(BM * BN / TM / TN);
    sgemm_thread_tiling_2d<BM, BK, BN, TM, TN><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_6_sgemm_thread_register(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
    dim3 grid_dim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_dim(BM * BN / TM / TN);
    sgemm_thread_register<BM, BK, BN, TM, TN><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_7_sgemm_vectorized_access(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
    dim3 grid_dim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_dim(BM * BN / TM / TN);
    sgemm_vectorized_access<BM, BK, BN, TM, TN><<<grid_dim, block_dim>>>(m, n, k, alpha, a, b, beta, c);
}

void run_cublas_sgemm(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b, n, a, k, &beta, c, n);
    cublasDestroy(handle);
}

void run_sgemm(
    int kernel_id,
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    if (kernel_id == 0)
        run_cublas_sgemm(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 1)
        run_1_sgemm_naive(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 2)
        run_2_sgemm_global_mem_coalescing(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 3)
        run_3_sgemm_shared_mem_caching(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 4)
        run_4_sgemm_thread_tiling_1d(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 5)
        run_5_sgemm_thread_tiling_2d(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 6)
        run_6_sgemm_thread_register(m, n, k, alpha, a, b, beta, c);
    else if (kernel_id == 7)
        run_7_sgemm_vectorized_access(m, n, k, alpha, a, b, beta, c);
}
