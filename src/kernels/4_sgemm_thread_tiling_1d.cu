#include <cuda_runtime.h>
#include <cassert>
#include "sgemm.cuh"


template <const int BM, const int BK, const int BN, const int TM>
__global__ void sgemm_thread_tiling_1d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];
    float thread_result[TM] = {0.0};

    const int bx = blockIdx.y;
    const int by = blockIdx.x;
    const int tx = threadIdx.x / BN;
    const int ty = threadIdx.x % BN;

    a = &a[bx * BM * k];
    b = &b[by * BN];
    c = &c[bx * BM * n + by * BN];

    const int tile_a_x = threadIdx.x / BK;
    const int tile_a_y = threadIdx.x % BK;
    const int tile_b_x = threadIdx.x / BN;
    const int tile_b_y = threadIdx.x % BN;

    for (int i = 0; i < k; i += BK) {
        tile_a[tile_a_x * BK + tile_a_y] = a[tile_a_x * k + tile_a_y];
        tile_b[tile_b_x * BN + tile_b_y] = b[tile_b_x * n + tile_b_y];
        __syncthreads();
        
        a += BK;
        b += BK * n;

        for (int j = 0; j < BK; j++) {
            float b_value = tile_b[j * BN + ty];
            for (int l = 0; l < TM; l++) {
                thread_result[l] += tile_a[(tx * TM + l) * BK + j] * b_value;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        c[(tx * TM + i) * n + ty] = alpha * thread_result[i] + beta * c[(tx * TM + i) * n + ty];
    }
}

template __global__ void sgemm_thread_tiling_1d<64, 8, 64, 8>(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);
