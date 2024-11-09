#ifndef UTILS_CUH
#define UTILS_CUH

#include <vector>
#include <cuda_runtime.h>


void cudaCheck(cudaError_t err, const char *file, int line);
void randomize_matrix(std::vector<float> &a);
std::vector<float> create_matrix(int m, int n);
void print_matrix(std::vector<float> &a, int m, int n, int limit);

class GPUTimer {
private:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    GPUTimer();
    ~GPUTimer();
    void start_timer();
    void stop_timer();
    float get_elapsed_time();
};

#endif
