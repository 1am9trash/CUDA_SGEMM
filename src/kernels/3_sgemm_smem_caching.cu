#include <cuda_runtime.h>
#include "sgemm.cuh"


template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_caching(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    __shared__ float tile_a[BLOCKSIZE * BLOCKSIZE];
    __shared__ float tile_b[BLOCKSIZE * BLOCKSIZE];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int x = threadIdx.x / BLOCKSIZE;
    const int y = threadIdx.x % BLOCKSIZE;

    a = &a[bx * BLOCKSIZE * k];
    b = &b[by * BLOCKSIZE];
    c = &c[bx * BLOCKSIZE * n + by * BLOCKSIZE];

    float value = 0.0;
    for (int i = 0; i < k; i += BLOCKSIZE) {
        tile_a[x * BLOCKSIZE + y] = a[x * k + y];
        tile_b[x * BLOCKSIZE + y] = b[x * n + y];
        __syncthreads();
        
        a += BLOCKSIZE;
        b += BLOCKSIZE * n;

        for (int j = 0; j < BLOCKSIZE; j++) {
            value += tile_a[x * BLOCKSIZE + j] * tile_b[j * BLOCKSIZE + y];
        }
        __syncthreads();
    }
    c[x * n + y] = alpha * value + beta * c[x * n + y];
}

template __global__ void sgemm_shared_mem_caching<32>(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);
