#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "kernel_runner.cuh"
#include "utils.hpp"


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main() {
    int kernel_id = 1;

    const int device_id = 0;
    cudaCheck(cudaSetDevice(device_id));
    std::cout << "running kernel on device " << device_id << "\n\n";

    cudaDeviceProp device_prop;
    cudaCheck(cudaGetDeviceProperties(&device_prop, device_id));
    std::cout << "shared memory per SM: " << device_prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "shared memory per block: " << device_prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "warp size: " << device_prop.warpSize << "\n\n";

    const std::vector<int> test_size = {4096};
    const int max_test_size = test_size[test_size.size() - 1];
    int m, k, n;
    const float alpha = 0.1, beta = 0.2;
    std::vector<float> host_a, host_b, host_c;
    float *device_a, *device_b, *device_c;

    host_a = create_matrix(max_test_size, max_test_size);
    host_b = create_matrix(max_test_size, max_test_size);
    host_c = create_matrix(max_test_size, max_test_size);

    const int byte_count = sizeof(float) * host_a.size();
    cudaCheck(cudaMalloc((void **)&device_a, byte_count));
    cudaCheck(cudaMalloc((void **)&device_b, byte_count));
    cudaCheck(cudaMalloc((void **)&device_c, byte_count));
    cudaCheck(cudaMemcpy(device_a, host_a.data(), byte_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_b, host_b.data(), byte_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_c, host_c.data(), byte_count, cudaMemcpyHostToDevice));

    const int test_count = 50;
    GPUTimer timer;
    for (int i = 0; i < test_size.size(); i++) {
        m = n = k = test_size[i];
        timer.start_timer();
        for (int j = 0; j < test_count; j++) {
            run_sgemm(kernel_id, m, n, k, alpha, device_a, device_b, beta, device_c);
        }
        timer.stop_timer();

        float elapsed_time = timer.get_elapsed_time() / 1000;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "dimensions (m=k=n): " << m << ", alpha: " << alpha << ", beta: " << beta << "\n";
        std::cout << "elapsed time: " << elapsed_time / test_count * 1000 << " ms\n";
        std::cout << "performance: " << ((double)test_count * 2 * m * n * k * 1e-9) / elapsed_time << " GFLOPS\n";

        cudaCheck(cudaMemcpy(host_c.data(), device_c, byte_count, cudaMemcpyDeviceToHost));
        std::cout << "part of matrix:\n";
        print_matrix(host_c, m, n, 8);
        std::cout << "\n";
    }

    return 0;
}
