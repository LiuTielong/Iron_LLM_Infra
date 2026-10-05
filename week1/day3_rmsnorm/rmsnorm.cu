#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdlib>


// 在CPU上面执行的 RMSNorm 函数
std::vector<float> rmsnorm_cpu(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    int rows, 
    int hidden_size,
    float eps
){
    // 输出数组，与输入大小相同
    std::vector<float> y(x.size());
    
    for (int row = 0; row < rows; row++){
        // 当前行在一堆数组中的起始位置
        std::size_t base = static_cast<std::size_t>(row) * hidden_size;

        // 1. 计算当前行的平方和
        double sum_sq = 0.0;

        for (int j = 0; j < hidden_size; j++){
            double value = static_cast<double>(x[base + j]);
            sum_sq += value * value;
        }

        // 2. 计算RMS的倒数
        double mean_sq = sum_sq / hidden_size;
        double inv_rms = 1.0 / std::sqrt(mean_sq + eps);

        // 3. 归一化，再乘以每个位置的权重
        for (int j = 0; j < hidden_size; j++){
            y[base + j] = static_cast<float>(
                x[base + j] * inv_rms * weight[j]);
        }
    }

    return y;
}

// 基础版kernel。一行对应一个Block。
__global__ void rmsnorm_naive(
    const float* x,
    const float* weight,
    float* y,
    int rows,
    int hidden_size,
    float eps
){ 
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows){
        return;
    }

    std::size_t base = static_cast<std::size_t>(row) * hidden_size;

    // 每个线程累加自己负责的元素
    // 比如，block size=256，而一行有1024个数据，那么每个线程负责4个数据
    float thread_sum = 0.0f;

    for (int j = tid; j < hidden_size; j += blockDim.x){
        thread_sum += x[base + j] * x[base + j];
    }


    // 保存每个线程的局部平方和
    sdata[tid] = thread_sum;
    __syncthreads();

    // block 内规约平方和
    // 是block级别的协作，但归根结底还是每个线程在执行。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // sdata[0] 是当前行的平方和
    float mean_sq = sdata[0] / static_cast<float>(hidden_size);

    float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    // 第二次遍历，获得归一化结果
    // 每个线程负责4个 y 的生成
    for (int j = tid; j < hidden_size; j += blockDim.x){
        y[base + j] = x[base + j] * inv_rms * weight[j];
    }
}


int main(){
    int rows = 128;
    int hidden_size = 1024;
    float eps = 1e-5f;

    std::size_t num_elements = static_cast<std::size_t>(rows) * hidden_size;

    // 1. 数据准备
    std::vector<float> h_x(num_elements);  // 输入
    std::vector<float> h_weight(hidden_size);   // 权重

    for (std::size_t i=0; i < num_elements; i++){
        h_x[i] = 0.02f * (static_cast<int>(i % 101) - 50);
    }

    for (std::size_t i = 0; i < hidden_size; i++){
        h_weight[i] = 1.0f + 0.01f * (i % 7);
    }

    // 2. 得到CPU参考输出，后续与GPU输出比较
    std::vector<float> h_ref = rmsnorm_cpu(h_x, h_weight, rows, hidden_size, eps);


    // 3. GPU 部分
    float* d_x = nullptr;
    float* d_weight = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_x, num_elements * sizeof(float));
    cudaMalloc(&d_weight, hidden_size * sizeof(float));
    cudaMalloc(&d_y, num_elements * sizeof(float));

    cudaMemcpy(
        d_x, 
        h_x.data(), 
        num_elements * sizeof(float), 
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_weight, 
        h_weight.data(), 
        hidden_size * sizeof(float), 
        cudaMemcpyHostToDevice
    );

    int block_size = 128;

    std::size_t shared_bytes = block_size * sizeof(float);

    rmsnorm_naive<<<rows, block_size, shared_bytes>>>(
        d_x,
        d_weight,
        d_y,
        rows,
        hidden_size,
        eps
    );

    cudaDeviceSynchronize();


    // 4. 功能测试：比较CPU和GPU的输出
    std::vector<float> h_y(num_elements);

    cudaMemcpy(
        h_y.data(),
        d_y,
        num_elements * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // 比较所有元素
    float max_error = 0.0f;

    for (std::size_t i = 0; i < num_elements; ++i) {
        float error = std::fabs(h_y[i] - h_ref[i]);

        // 防止 NaN/Inf 被误认为没有误差
        if (!std::isfinite(error)) {
            std::cout << "结果出现 NaN/Inf\n";
            return 1;
        }

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << std::scientific
            << "max_abs_error = " << max_error << "\n";

    // 对当前这组 FP32 输入，先用此阈值验收
    if (max_error > 1e-5f) {
        std::cout << "功能测试未通过\n";
        return 1;
    }

    std::cout << "功能测试通过\n";


    // 5. 性能测试
    for (int i = 0; i < 20; ++i) {
        rmsnorm_naive<<<rows, block_size, shared_bytes>>>(
            d_x, d_weight, d_y, rows, hidden_size, eps
        );
    }

    cudaDeviceSynchronize();

    // 创建计时 Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int repeats = 100;

    cudaEventRecord(start);

    for (int i = 0; i < repeats; ++i) {
        rmsnorm_naive<<<rows, block_size, shared_bytes>>>(
            d_x, d_weight, d_y, rows, hidden_size, eps
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算平均时间
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    float avg_ms = total_ms / repeats;

    std::cout << std::fixed << std::setprecision(6)
            << "avg_kernel_ms = " << avg_ms << "\n";

    double bytes = static_cast<double>(2.0 * num_elements + hidden_size) * sizeof(float);
    double bandwidth_gbs = bytes / (avg_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(3)
            << "bandwidth_gbs = " << bandwidth_gbs << "\n";

    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_y);

    return 0;
}