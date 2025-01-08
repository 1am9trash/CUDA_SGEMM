#include <cuda_runtime.h>
#include <cassert>
#include "sgemm.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm_vectorized_access(
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
    float frag_a[TM];
    float frag_b[TN];

    const int bx = blockIdx.y;
    const int by = blockIdx.x;
    const int tx = (threadIdx.x / (BN / TN)) * TM;
    const int ty = (threadIdx.x % (BN / TN)) * TN;

    a = &a[bx * BM * k];
    b = &b[by * BN];
    c = &c[bx * BM * n + by * BN];

    const int tile_a_x = threadIdx.x / (BK / 4);
    const int tile_a_y = threadIdx.x % (BK / 4) * 4;
    const int tile_load_a_iter = (BM * BK) / (BM * BN / (TM * TN)) / 4;
    const int tile_a_stride = BM / tile_load_a_iter;
    const int tile_b_x = threadIdx.x / (BN / 4);
    const int tile_b_y = threadIdx.x % (BN / 4) * 4;
    const int tile_load_b_iter = (BK * BN) / (BM * BN / (TM * TN)) / 4;
    const int tile_b_stride = BK / tile_load_b_iter;

    for (int i = 0; i < k; i += BK) {
        for (int offset = 0; offset < BM; offset += tile_a_stride) {
            float4 tmp = FETCH_FLOAT4(&a[(tile_a_x + offset) * k + tile_a_y]);
            tile_a[(tile_a_y + offset) * BM + tile_a_x] = tmp.x;
            tile_a[(tile_a_y + offset + 1) * BM + tile_a_x] = tmp.y;
            tile_a[(tile_a_y + offset + 2) * BM + tile_a_x] = tmp.z;
            tile_a[(tile_a_y + offset + 3) * BM + tile_a_x] = tmp.w;
        }
        for (int offset = 0; offset < BK; offset += tile_b_stride) {
            FETCH_FLOAT4(&tile_b[(tile_b_x + offset) * BN + tile_b_y]) = FETCH_FLOAT4(&b[(tile_b_x + offset) * n + tile_b_y]);
        }
        __syncthreads();
        
        a += BK;
        b += BK * n;

        for (int j = 0; j < BK; j++) {
            for (int id_m = 0; id_m < TM; id_m += 4) {
                FETCH_FLOAT4(&frag_a[id_m]) = FETCH_FLOAT4(&tile_a[j * BM + tx + id_m]);
            }
            for (int id_n = 0; id_n < TN; id_n += 4) {
                FETCH_FLOAT4(&frag_b[id_n]) = FETCH_FLOAT4(&tile_b[j * BN + ty + id_n]);
            }
            for (int id_m = 0; id_m < TM; id_m++) {
                for (int id_n = 0; id_n < TN; id_n++) {
                    thread_result[id_m * TN + id_n] += frag_a[id_m] * frag_b[id_n];
                }
            }
        }
        __syncthreads();
    }

    for (int id_m = 0; id_m < TM; id_m++) {
        for (int id_n = 0; id_n < TN; id_n += 4) {
            float4 tmp = FETCH_FLOAT4(&c[(tx + id_m) * n + ty + id_n]);
            tmp.x = alpha * thread_result[id_m * TN + id_n] + beta * tmp.x;
            tmp.y = alpha * thread_result[id_m * TN + id_n + 1] + beta * tmp.y;
            tmp.z = alpha * thread_result[id_m * TN + id_n + 2] + beta * tmp.z;
            tmp.w = alpha * thread_result[id_m * TN + id_n + 3] + beta * tmp.w;
            FETCH_FLOAT4(&c[(tx + id_m) * n + ty + id_n]) = tmp;
        }
    }
}

template __global__ void sgemm_vectorized_access<128, 8, 128, 8, 8>(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);
