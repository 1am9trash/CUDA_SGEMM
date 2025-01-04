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

void print_GPU_info(int device_id) {
    cudaDeviceProp device_prop;
    cudaCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);

    std::cout << "\n";
    std::cout << "number of SMs: " << device_prop.multiProcessorCount << "\n";

    std::cout << "gmem size: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "smem per SM: " << device_prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "smem per block: " << device_prop.sharedMemPerBlock / 1024 << " KB\n";

    std::cout << "warp size: " << device_prop.warpSize << "\n";
    std::cout << "max threads per SM: " << device_prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "max threads per block: " << device_prop.maxThreadsPerBlock << "\n";
    std::cout << "max warps per SM: " << device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize << "\n";
    std::cout << "registers per block: " << device_prop.regsPerBlock / 1024 << " K\n";
    std::cout << "registers per SM: " << device_prop.regsPerMultiprocessor / 1024 << " K\n";

    std::cout << "max block dimensions: ("
              << device_prop.maxThreadsDim[0] << ", "
              << device_prop.maxThreadsDim[1] << ", "
              << device_prop.maxThreadsDim[2] << ")\n";
    std::cout << "max grid dimensions: ("
              << device_prop.maxGridSize[0] << ", "
              << device_prop.maxGridSize[1] << ", "
              << device_prop.maxGridSize[2] << ")\n";

    std::cout << "clock rate: " << device_prop.clockRate / 1000.0f << " MHz\n";
    std::cout << "memory clock rate: " << device_prop.memoryClockRate / 1000.0f << " MHz\n";
    std::cout << "memory bus width: " << device_prop.memoryBusWidth << " bits\n";
    std::cout << "compute capability: " << device_prop.major << "." << device_prop.minor << "\n";
    std::cout << "\n";
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
    std::cout << "\n";
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
