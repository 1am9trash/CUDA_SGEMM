#include <cuda_runtime.h>
#include "sgemm.cuh"


__global__ void sgemm_naive(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < m) && (col < n)) {
        float value = 0.0;
        for (int i = 0; i < k; i++) {
            value += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = alpha * value + beta * c[row * n + col];
    }
}
