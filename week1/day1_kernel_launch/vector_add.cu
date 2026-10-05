#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    // 计算每个线程负责的全局数组下标
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(){
    // 1. 设置输入规模
    int n = (1 << 20) + 13;
    std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);   // std::size_t 是 C++ 中常用于表示内存大小和数组下标的无符号整数类型

    // 2. 准备Host端数据
    std::vector<float> h_a(n);  
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f + 0.001f * i;   // 后缀f表示这是一个float常量，而不是默认的double。
        h_b[i] = 2.0f - 0.0005f * i;
    }

    // 3. 准备 Device 指针
    float* d_a = nullptr;          
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, bytes);        // 在GPU上分配bytes字节的显存，并把地址写入d_a. 
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 4. Host --> Device
    cudaMemcpy( 
        d_a,                        // 目标地址
        h_a.data(),                 // 源地址：h_a.data() 返回 vector 内部连续存储区域的首地址
        bytes,                      // 数据量
        cudaMemcpyHostToDevice      // 拷贝方向
    );

    cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    // 5. 设置Grid 和 Block
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // 启动 kernel
    // CUDA 特有的 Kernel Launch 语法
    vector_add<<<grid_size, block_size>>>(
        d_a,
        d_b,
        d_c,
        n
    );

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // Device -> Host
    cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // 检查结果
    float max_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        float error = std::fabs(h_c[i] - (h_a[i] + h_b[i]));

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "n = " << n << "\n";
    std::cout << "block_size = " << block_size << "\n";
    std::cout << "grid_size = " << grid_size << "\n";
    std::cout << "max_error = " << max_error << "\n";

    // 释放 GPU 显存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}