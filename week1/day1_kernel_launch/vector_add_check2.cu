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
    int block_size = 128;
    int grid_size = (n + block_size - 1) / block_size;

    // 启动 kernel
    cudaEvent_t start;  // 声明一个cudaEvent 句柄
    cudaEvent_t stop;

    cudaEventCreate(&start); // 创建Event对象
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        vector_add<<<grid_size, block_size>>>(
            d_a,
            d_b,
            d_c,
            n
        );
    }

    cudaDeviceSynchronize();

    // 正式计时
    int repeats = 100;

    cudaEventRecord(start);     // 在当前cudaEvent 中记录时间点

    for (int i = 0; i < repeats; ++i) {
        vector_add<<<grid_size, block_size>>>(
            d_a,
            d_b,
            d_c,
            n
        );
    }

    cudaEventRecord(stop);

    // 等待 stop Event 完成
    cudaEventSynchronize(stop);   // 等待 stop Event 所在位置之前的 GPU 工作全部完成。

    float total_ms = 0.0f;

    cudaEventElapsedTime(           // 计算两个Event之间的GPU时间，单位是毫秒
        &total_ms,
        start,
        stop
    );

    // 计算执行时间
    float avg_ms = total_ms / repeats;

    std::cout << "total_kernel_ms = "
            << total_ms << "\n";

    std::cout << "avg_kernel_ms = "
            << avg_ms << "\n";


    // 计算有效带宽
    double total_bytes =
        3.0 * n * sizeof(float);

    double bandwidth_gbs =
        total_bytes / (avg_ms * 1e6);

    std::cout << "bandwidth_gbs = "
            << bandwidth_gbs << "\n";

    // Device -> Host
    cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // 释放 GPU 显存
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}