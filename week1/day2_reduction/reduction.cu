
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include <iomanip>


// atomicAdd 保证更新是原子的，但所有线程竞争同一个地址，可能导致严重的串行化。
__global__ void reduce_atomic(
    const float* x,     
    float* out, 
    int n
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        atomicAdd(out, x[i]);  // 原子加法函数
    }
}


// global 表示这是一个CUDA kernel，由CPU端调用，在GPU上执行。
__global__ void reduce_shared(
    const float* x,     // x 是输入数组的GPU地址，const 表示 kernel不会修改 x 中的数据
    float* out,         // 输出地址，所有block最终把自己的局部结果加到这里
    int n               // 输入数组的长度
){
    // 声明一块动态分配的 shared memory. 每个block都有自己独立的sdata. 同一block内的线程可以访问它
    // 通常每个线程负责一个 shared memory 元素
    extern __shared__ float sdata[];     

    // 获取线程在线程块中的编号
    // 然后计算线程的全局数组下标
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 把输入加载到shared memory
    if (i<n) {
        sdata[tid] = x[i];
    } else {
        sdata[tid] = 0.0f;
    }

    // 第一次同步
    // 这是在block内部的线程同步
    __syncthreads();

    // 开始树形归约
    for (
        int stride = blockDim.x / 2;
        stride >0;
        stride >>= 1
    ){
        // 正式进行归约计算
        // 每一轮，把后面的线程中的数据加到前面的线程的shared memory中。
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        // 每轮归约后的同步
        __syncthreads();
    }
    
    // 经过上面的计算之后，每个block的线程0都有一个部分和。
    // 把每个block的结果写入全局输出
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }

}


// 添加warp内规约函数
// __device__ 表示这个函数只能由GPU调用。
__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>=1) {
        // 这个函数表示当前线程从同一个warp中，编号更大的另一个线程中读取一个寄存器值。
        // 不断减小offset，最终lane 0 得到整个warp的和。
        value += __shfl_down_sync(          
            0xffffffff,
            value,
            offset
        );
    }

    return value;
}

__global__ void reduce_warp_shuffle(
    const float* x,
    float* out,
    int n
){
    // 每个block一份shared memory，大小由kernel启动时的第三个参数决定。
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;

    int lane = tid % 32;
    int warp_id = tid / 32;

    int i = blockIdx.x * blockDim.x + tid;

    float value = 0.0f;

    if (i < n){
        value = x[i];
    }

    // 第一步：每个warp内规约
    value = warp_reduce_sum(value);

    // 每个warp 的 lane 0 保存该warp的局部和
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }

    // 确保所有warp都完成写入
    __syncthreads();

    int warp_count = (blockDim.x + 31) / 32;

    // 第二步：让第一个warp规约所有warp的局部和
    if (warp_id == 0){
        float block_sum = 0.0f;

        if (lane < warp_count) {            // 每个 Block 内部的第一个warp 的前8个线程，存储了这8个warp_sum。   
            block_sum = warp_sums[lane];    // 其他24个线程取0. 
        }

        block_sum = warp_reduce_sum(block_sum);  // 第一个warp的线程0得到整个block的综合。

        // 第一个warp的lane 0 得到整个block的结果
        if (lane == 0){
            atomicAdd(out, block_sum);
        }
    }
}


int main(){
    float* d_x;
    float* d_out;

    // host 端数据准备
    int n = 1 << 20;
    std::size_t input_bytes = n * sizeof(float);

    std::vector<float> h_x(n);

    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f + 0.001f * (i%13);
    }

    // 分配GPU显存：
    cudaMalloc(&d_x, input_bytes);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(
        d_x, 
        h_x.data(),
        input_bytes,
        cudaMemcpyHostToDevice
    );


    /*
    // 方法一：使用原子操作 reduce_atomic
    // 启动kernel
    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    for (int i = 0; i<20; i++){
        cudaMemset(d_out, 0, sizeof(float));

        reduce_atomic<<<grid_size, block_size>>>(
            d_x,
            d_out,
            n
        );
    }
    cudaDeviceSynchronize();

    // 正式计算
    const int repeats = 100;
    float total_ms = 0.0f;

    for (int i = 0; i < repeats; ++i) {
        cudaMemset(d_out, 0, sizeof(float));
        cudaDeviceSynchronize();

        cudaEventRecord(start);

        reduce_atomic<<<grid_size, block_size>>>(
            d_x,
            d_out,
            n
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        total_ms += elapsed_ms;
    }

    */


    /*
    // 方法二：使用shared memory，在每个block内部分别计算
    int block_size = 1024;
    int grid_size = (n + block_size -1) / block_size;

    std::size_t shared_memory_bytes = block_size * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    for (int i = 0; i<20; i++){
        cudaMemset(d_out, 0, sizeof(float));

        reduce_shared<<<grid_size, block_size, shared_memory_bytes>>>(
            d_x,
            d_out,
            n
        );
    }
    cudaDeviceSynchronize();

    // 正式计算
    const int repeats = 100;
    float total_ms = 0.0f;

    for (int i = 0; i < repeats; i++){
        cudaMemset(d_out, 0, sizeof(float));
        cudaDeviceSynchronize();

        cudaEventRecord(start);

        reduce_shared<<<grid_size, block_size, shared_memory_bytes>>>(
            d_x,
            d_out,
            n
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        total_ms += elapsed_ms;
    }
    */

    // 方法3：在warp内规约
    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;
    int warp_count = (block_size + 31) / 32;

    std::size_t shared_memory_bytes = warp_count * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    for (int i = 0; i<200; i++){
        cudaMemset(d_out, 0, sizeof(float));

        reduce_warp_shuffle<<<grid_size, block_size, shared_memory_bytes>>>(
            d_x, 
            d_out,
            n
        );
    }
    cudaDeviceSynchronize();

    // 正式计算
    const int repeats = 1000;
    float total_ms = 0.0f;

    for (int i = 0; i < repeats; i++){
        cudaMemset(d_out, 0, sizeof(float));
        cudaDeviceSynchronize();

        cudaEventRecord(start);

        reduce_warp_shuffle<<<grid_size, block_size, shared_memory_bytes>>>(
            d_x, 
            d_out,
            n
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        total_ms += elapsed_ms;
    }

    float avg_ms = total_ms / repeats;

    std::cout << "avg_kernel_ms = "
            << avg_ms << "\n";

    double bytes = static_cast<double>(n) * sizeof(float);
    double bandwidth_gbs = bytes / (avg_ms * 1.0e6);

    std::cout << "bandwidth_gbs = "
            << bandwidth_gbs << "\n";

    // 把标量结果写回CPU
    float gpu_sum = 0.0f;

    cudaMemcpy(
        &gpu_sum,
        d_out,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // 验证结果
    // 先写一个CPU的结果，然后验证。
    // float cpu_sum = 0.0f;

    // for (int i = 0; i < n; ++i) {
    //     cpu_sum += h_x[i];
    // }   

    // float error = std::fabs(gpu_sum - cpu_sum);
    
    // std::cout << "cpu_sum = " << cpu_sum << "\n";
    // std::cout << "gpu_sum = " << gpu_sum << "\n";
    // std::cout << "error = " << error << "\n";

    // 释放显存
    cudaFree(d_x);
    cudaFree(d_out);

}
