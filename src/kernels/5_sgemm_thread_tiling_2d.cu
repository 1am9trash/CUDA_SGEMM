#include <cuda_runtime.h>
#include <cassert>
#include "sgemm.cuh"


template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm_thread_tiling_2d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    assert(BM * BN / TM / TN == blockDim.x);
    assert(BM % TM == 0);
    assert(BN % TN == 0);
    assert((BM * BN / TM / TN) % BK == 0);
    assert((BM * BN / TM / TN) % BN == 0);

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];
    float thread_result[TM * TN] = {0.0};

    const int bx = blockIdx.y;
    const int by = blockIdx.x;
    const int tx = (threadIdx.x / (BN / TN)) * TM;
    const int ty = (threadIdx.x % (BN / TN)) * TN;

    a = &a[bx * BM * k];
    b = &b[by * BN];
    c = &c[bx * BM * n + by * BN];

    const int tile_a_x = threadIdx.x / BK;
    const int tile_a_y = threadIdx.x % BK;
    const int tile_a_stride = BM * BN / (TM * TN) / BK;
    const int tile_b_x = threadIdx.x / BN;
    const int tile_b_y = threadIdx.x % BN;
    const int tile_b_stride = BM * BN / (TM * TN) / BN;

    for (int i = 0; i < k; i += BK) {
        for (int offset = 0; offset < BM; offset += tile_a_stride) {
            tile_a[(tile_a_x + offset) * BK + tile_a_y] = a[(tile_a_x + offset) * k + tile_a_y];
        }
        for (int offset = 0; offset < BK; offset += tile_b_stride) {
            tile_b[(tile_b_x + offset) * BN + tile_b_y] = b[(tile_b_x + offset) * n + tile_b_y];
        }
        __syncthreads();
        
        a += BK;
        b += BK * n;

        for (int j = 0; j < BK; j++) {
            for (int id_m = 0; id_m < TM; id_m++) {
                for (int id_n = 0; id_n < TN; id_n++) {
                    thread_result[id_m * TN + id_n] += tile_a[(tx + id_m) * BK + j] * tile_b[j * BN + ty + id_n];
                }
            }
        }
        __syncthreads();
    }

    for (int id_m = 0; id_m < TM; id_m++) {
        for (int id_n = 0; id_n < TN; id_n++) {
            c[(tx + id_m) * n + ty + id_n] = alpha * thread_result[id_m * TN + id_n] + beta * c[(tx + id_m) * n + ty + id_n];
        }
    }
}

template __global__ void sgemm_thread_tiling_2d<128, 8, 128, 8, 8>(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);
