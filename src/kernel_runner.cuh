#ifndef KERNEL_RUNNER_CUH
#define KERNEL_RUNNER_CUH

#include <cuda_runtime.h>


void run_1_sgemm_naive(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_sgemm(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

#endif
