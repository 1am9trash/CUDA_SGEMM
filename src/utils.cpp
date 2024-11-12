#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include "utils.hpp"


void cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << ".\n";
        exit(EXIT_FAILURE);
    }
}

void randomize_matrix(std::vector<float> &a) {
    static std::default_random_engine rng(1337);
    static std::uniform_real_distribution<float> dist;
    std::generate(a.begin(), a.end(), [&]{ return dist(rng); });
}

std::vector<float> create_matrix(int m, int n) {
    std::vector<float> a(m * n);
    randomize_matrix(a);
    return a;
}

void print_matrix(std::vector<float> &a, int m, int n, int limit) {
    for (int i = 0; i < std::min(limit, m); i++) {
        for (int j = 0; j < std::min(limit, n); j++) {
            std::cout << a[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

bool is_matrix_same(std::vector<float> &a, std::vector<float> &b) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        if (fabs(a[i] - b[i]) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

GPUTimer::GPUTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GPUTimer::start_timer() {
    cudaEventRecord(start, 0);
}

void GPUTimer::stop_timer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
}

float GPUTimer::get_elapsed_time() {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}
