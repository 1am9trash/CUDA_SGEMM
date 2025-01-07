#ifndef KERNEL_RUNNER_CUH
#define KERNEL_RUNNER_CUH

#include <cuda_runtime.h>


void run_1_sgemm_naive(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_2_sgemm_global_mem_coalescing(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_3_sgemm_shared_mem_caching(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_4_sgemm_thread_tiling_1d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_5_sgemm_thread_tiling_2d(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_6_sgemm_thread_register(
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

void run_sgemm(
    int kernel_id,
    int m, int n, int k,
    const float alpha, const float *a, const float *b, const float beta,
    float *c
);

#endif
