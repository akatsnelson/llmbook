// Глава 15 «Устройство GPU», раздел 15.2 «Компиляция ядра CUDA»: полная программа с ядром axpy
// для формулы y(i) = a x(i) + b(i), с выделением памяти и проверкой результата.
// Нужны карта NVIDIA и компилятор nvcc. Сборка и запуск из каталога labs/15_cuda_path:
//   nvcc -O3 axpy.cu -o axpy && ./axpy        (печатает ok: y[0]=10.0, ...)
// Остальные шаги лабораторной (PTX, машинный код, профиль) описаны в README.md рядом.

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

__global__ void axpy(const float* x, const float* b,
                     float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + b[i];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr float a = 3.0f;
    std::vector<float> x(n, 2.0f);
    std::vector<float> b(n, 4.0f);
    std::vector<float> y(n, 0.0f);

    float *dx = nullptr, *db = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    axpy<<<blocks, threads>>>(dx, db, dy, a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (float value : y) {
        if (std::fabs(value - 10.0f) > 1e-6f) {
            std::fprintf(stderr, "wrong result: %.8f\n", value);
            return 2;
        }
    }
    std::printf("ok: y[0]=%.1f, elements=%d\n", y[0], n);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
