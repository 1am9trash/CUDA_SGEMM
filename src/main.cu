#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "kernel_runner.cuh"
#include "utils.hpp"


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main() {
    // ------

    int kernel_id, test_size;
    bool check_answer;
    std::cout << "kernel id = ";
    std::cin >> kernel_id;
    std::cout << "matrix size = ";
    std::cin >> test_size;
    std::cout << "check answer correctness or not (0/1) = ";
    std::cin >> check_answer;

    // ------

    const int device_id = 0;
    print_GPU_info(device_id);
    cudaCheck(cudaSetDevice(device_id));
    std::cout << "running kernel on device " << device_id << "\n\n";

    int m, k, n;
    const float alpha = 0.1, beta = 0.2;
    std::vector<float> host_a, host_b, host_c;
    float *device_a, *device_b, *device_c;

    host_a = create_matrix(test_size, test_size);
    host_b = create_matrix(test_size, test_size);
    host_c = create_matrix(test_size, test_size);

    const int byte_count = sizeof(float) * host_a.size();
    cudaCheck(cudaMalloc((void **)&device_a, byte_count));
    cudaCheck(cudaMalloc((void **)&device_b, byte_count));
    cudaCheck(cudaMalloc((void **)&device_c, byte_count));
    cudaCheck(cudaMemcpy(device_a, host_a.data(), byte_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_b, host_b.data(), byte_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_c, host_c.data(), byte_count, cudaMemcpyHostToDevice));

    // ------

    const int test_count = 10;
    GPUTimer timer;
    m = k = n = test_size;

    timer.start_timer();
    for (int i = 0; i < test_count; i++) {
        run_sgemm(kernel_id, m, n, k, alpha, device_a, device_b, beta, device_c);
    }
    timer.stop_timer();
    float elapsed_time = timer.get_elapsed_time() / 1000;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "dimensions (m=k=n): " << m << ", alpha: " << alpha << ", beta: " << beta << "\n";
    std::cout << "elapsed time: " << elapsed_time / test_count * 1000 << " ms\n";
    std::cout << "performance: " << ((double)test_count * 2 * m * n * k * 1e-9) / elapsed_time << " GFLOPS\n\n";

    // ------

    if (check_answer) {
        cudaCheck(cudaMemcpy(device_c, host_c.data(), byte_count, cudaMemcpyHostToDevice));
        run_sgemm(kernel_id, m, n, k, alpha, device_a, device_b, beta, device_c);
        std::vector<float> kernel_result = std::vector<float>(m * n);
        cudaCheck(cudaMemcpy(kernel_result.data(), device_c, byte_count, cudaMemcpyDeviceToHost));

        std::cout << "part of matrix:\n";
        print_matrix(kernel_result, m, n, 8);

        std::vector<float> ground_truth = std::vector<float>(m * n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float value = 0.0;
                for (int l = 0; l < k; l++) {
                    value += host_a[i * k + l] * host_b[l * n + j];
                }
                ground_truth[i * n + j] = alpha * value + beta * host_c[i * n + j];
            }
        }
        std::cout << "matrix correctness: " << is_matrix_same(ground_truth, kernel_result) << "\n\n"; 
    }

    cudaCheck(cudaFree(device_a));
    cudaCheck(cudaFree(device_b));
    cudaCheck(cudaFree(device_c));

    return 0;
}
