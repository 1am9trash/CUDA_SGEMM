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

#endif
