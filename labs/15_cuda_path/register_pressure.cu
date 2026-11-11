// Глава 15 «Устройство GPU», раздел 15.5 «Давление на регистры и занятость»: два ядра,
// с одной и с восемью частичными суммами, для опытов с флагом --maxrregcount.
// Нужны карта NVIDIA и компилятор nvcc. Сборка и запуск из каталога labs/15_cuda_path:
//   nvcc -O3 --ptxas-options=-v register_pressure.cu -o register_pressure && ./register_pressure
// Порядок опыта описан в README.md рядом.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(error_));                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__global__ void one_accumulator(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = x[i];
#pragma unroll 1
    for (int k = 0; k < 256; ++k) {
        acc = fmaf(acc, 1.000001f, 0.000001f * float(k + 1));
    }
    y[i] = acc;
}

__global__ void eight_accumulators(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a0 = x[i] + 0.00f;
    float a1 = x[i] + 0.01f;
    float a2 = x[i] + 0.02f;
    float a3 = x[i] + 0.03f;
    float a4 = x[i] + 0.04f;
    float a5 = x[i] + 0.05f;
    float a6 = x[i] + 0.06f;
    float a7 = x[i] + 0.07f;
#pragma unroll 1
    for (int k = 0; k < 32; ++k) {
        float c = 0.000001f * float(k + 1);
        a0 = fmaf(a0, 1.000001f, c);
        a1 = fmaf(a1, 1.000002f, c);
        a2 = fmaf(a2, 1.000003f, c);
        a3 = fmaf(a3, 1.000004f, c);
        a4 = fmaf(a4, 1.000005f, c);
        a5 = fmaf(a5, 1.000006f, c);
        a6 = fmaf(a6, 1.000007f, c);
        a7 = fmaf(a7, 1.000008f, c);
    }
    y[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

float time_kernel(const void* kernel, const float* x, float* y, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void* arguments[] = {&x, &y, &n};
    CUDA_CHECK(cudaLaunchKernel(kernel, dim3(blocks), dim3(threads),
                                arguments, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < 100; ++repeat) {
        CUDA_CHECK(cudaLaunchKernel(kernel, dim3(blocks), dim3(threads),
                                    arguments, 0, nullptr));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds / 100.0f;
}

int main() {
    constexpr int n = 1 << 20;
    std::vector<float> host(n, 0.5f);
    float *x = nullptr, *y = nullptr;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float one_ms = time_kernel((const void*)one_accumulator, x, y, n);
    float eight_ms = time_kernel((const void*)eight_accumulators, x, y, n);
    CUDA_CHECK(cudaMemcpy(host.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(host[0])) return 2;
    std::printf("one_accumulator_ms=%.6f\n", one_ms);
    std::printf("eight_accumulators_ms=%.6f\n", eight_ms);
    std::printf("Do not compare times without ptxas and profiler reports.\n");

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 0;
}
