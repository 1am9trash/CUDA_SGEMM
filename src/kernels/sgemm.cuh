#ifndef SGEMM_CUH
#define SGEMM_CUH

#include <cuda_runtime.h>


/* 
sgemm computation: 
c = alpha * (a @ b) + beta * c

m: number of rows of matrix a and c
k: number of columns of matrix a and rows of c
n: number of columns of matrix b and c
*/

__global__ void sgemm_naive(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

template <const int BLOCKSIZE>
__global__ void sgemm_global_mem_coalescing(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_caching(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

template <const int BM, const int BK, const int BN, const int TM>
__global__ void sgemm_thread_tiling_1d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm_thread_tiling_2d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

#endif
