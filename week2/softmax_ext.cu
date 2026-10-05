#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cmath>

// 每个bloc负责一行，每个线程负责该行的多个元素
__global__ void softmax_kernel(
    const float* x,
    float* y,
    int rows,
    int cols
){ 
    // 索引形式
    int row = blockIdx.x;
    if (row >= rows){
        return;
    }
    int tid = threadIdx.x;
    std::size_t base = static_cast<std::size_t>(row) * cols;

    // shared memory 声明
    extern __shared__ float sdata[];

    // Stage 1: 计算行最大值
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[base + i];
        max_val = fmaxf(val, max_val);
    }
    sdata[tid] = max_val;
    __syncthreads();


    // 块内归约，最后sdata[0]保存了该行最大值
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + i]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];  // 线程自己的局部变量
    __syncthreads();

    // Stage 2：计算指数和
    // 重新让每个线程遍历自己的元素
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(x[base + i] - row_max);
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // 块内归约，最后sdata[0]保存了该行指数和
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    // Stage 3: 写出结果
    for (int i = tid; i < cols; i += blockDim.x) {
        y[base + i] = expf(x[base + i] - row_max) / row_sum;
    }
}

// 使用CUB库，调用归约
__global__ void softmax_kernel_cub(
    const float* x,
    float* y,
    int rows,
    int cols
){ 
    // 索引形式
    int row = blockIdx.x;
    if (row >= rows){
        return;
    }
    int tid = threadIdx.x;
    std::size_t base = static_cast<std::size_t>(row) * cols;

    // CUB 的归约配置：float 类型，每个 Block 256 个线程
    using BlockReduce = cub::BlockReduce<float, 256>;

    // 每个 Block 一份临时空间和归约结果
    __shared__ BlockReduce::TempStorage temp_storage;
    __shared__ float row_max;
    __shared__ float row_sum;

    // Stage 1：每个线程计算局部最大值
    float local_max = -FLT_MAX;

    for (int j = tid; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, x[base + j]);
    }

    // CUB 完成 Block 内最大值归约
    float block_max =
        BlockReduce(temp_storage).Reduce(local_max, cub::Max());

    // 最终结果只在线程 0 有效，由它写入共享变量
    if (tid == 0) {
        row_max = block_max;
    }

    // 确保所有线程能读取 row_max，
    // 同时确保下一次归约可以安全复用 temp_storage
    __syncthreads();

    // Stage 2：每个线程计算局部指数和
    float local_sum = 0.0f;

    for (int j = tid; j < cols; j += blockDim.x) {
        local_sum += expf(x[base + j] - row_max);
    }

    // CUB 完成 Block 内求和归约
    float block_sum =
        BlockReduce(temp_storage).Sum(local_sum);

    if (tid == 0) {
        row_sum = block_sum;
    }

    __syncthreads();

    // Stage 3：归一化并写出结果
    for (int j = tid; j < cols; j += blockDim.x) {
        y[base + j] =
            expf(x[base + j] - row_max) / row_sum;
    }
}



torch::Tensor softmax_forward(torch::Tensor x) {
    // 1. 检查输入，当前只支持非空、连续的二维 FP32 CUDA Tensor
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be a FP32 tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x must be non-empty");

    // 2. 选择输入所在的GPU
    c10::cuda::CUDAGuard guard(x.device());

    // 3. 分配输出，形状、dtype 和设备与 x 一致
    auto y = torch::empty_like(x);

    // 4. 配置：每个block 负责一行
    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));
    int block_size = 256;

    std::size_t shared_bytes = block_size * sizeof(float);

    // 5. 使用Pytorch 当前的 CUDA Stream
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    softmax_kernel<<<rows, block_size, shared_bytes, stream.stream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        cols
    );

    return y;
}

// 6. 暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_forward);
}