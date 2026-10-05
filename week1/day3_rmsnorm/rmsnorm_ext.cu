#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

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


// 追加接口
torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    // 本版只接受同一 GPU 上的连续 FP32 Tensor
    TORCH_CHECK(x.is_cuda() && weight.device() == x.device(),
                "x and weight must be on the same CUDA device");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32,
                "only float32 is supported");

    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "inputs must be contiguous");

    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1 &&
                x.size(0) > 0 && x.size(1) > 0 &&
                weight.size(0) == x.size(1),
                "expected x=[rows, hidden], weight=[hidden]");

    // 使用输入所在的 GPU
    c10::cuda::CUDAGuard guard(x.device());

    // PyTorch 分配输出显存
    auto y = torch::empty_like(x);

    int rows = static_cast<int>(x.size(0));
    int hidden_size = static_cast<int>(x.size(1));
    int block_size = 512;

    std::size_t shared_bytes = block_size * sizeof(float);

    // 接到 PyTorch 当前使用的 CUDA Stream 上
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    rmsnorm_naive<<<rows, block_size, shared_bytes, stream.stream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        hidden_size,
        static_cast<float>(eps)
    );

    return y;
}

// 将 C++ 函数暴露为 Python 模块中的 forward
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward);
}