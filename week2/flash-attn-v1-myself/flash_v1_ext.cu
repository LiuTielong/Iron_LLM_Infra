#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace flash_config {
    constexpr int HEAD_DIM = 128;
    constexpr int QUERY_TILE = 16;
    constexpr int KV_TILE = 128;

    constexpr int NUM_SPLITS = 1;
}

// 1. 从CPU传给GPU kernel的参数
struct FlashFwdParams{
    const void* q;
    const void* k;
    const void* v;
    void* o;

    int B, S, H, D;
    float scale;
};


// 2. GPU kernel函数
__global__ void flash_fwd_kernel(FlashFwdParams params){ 
    // TODO: Q/K/V 分块加载
    // TODO: QK^T、causal mask、Online Softmax、PV
    // TODO: 写回输出
}

// 3. CPU 接口：检查 Tensor、准备参数、启动 kernel
torch::Tensor flash_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
){
    TORCH_CHECK(
        q.is_cuda() && k.is_cuda() && v.is_cuda(), 
        "q, k, v must be CUDA tensors"
    );
    TORCH_CHECK(
        q.device() == k.device() && k.device() == v.device(),
        "q, k, v must be on the same device"
    );
    TORCH_CHECK(
        q.dim() == 4, "expected layout [B, S, H, D]"
    );
    TORCH_CHECK(
        q.sizes() == k.sizes() && k.sizes() == v.sizes(),
        "q, k, v must have the same shape"
    );
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "q, k, v must be float16 or bfloat16"
    );
    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
        "q, k, v must have the same dtype"
    );
    TORCH_CHECK(
        q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "q, k, v must be contiguous"
    );
    TORCH_CHECK(
        q.size(3) == 128, "head dim must be 128"
    );
    TORCH_CHECK(
        q.numel() > 0, "input must not be empty"
    );

    // 确保在输入所在的GPU上执行
    c10::cuda::CUDAGuard device_guard(q.device());

    auto out = torch::empty_like(q);
    
    FlashFwdParams params{};
    params.q = q.data_ptr();
    params.k = k.data_ptr();
    params.v = v.data_ptr();
    params.o = out.data_ptr();

    params.B = static_cast<int>(q.size(0));
    params.S = static_cast<int>(q.size(1));
    params.H = static_cast<int>(q.size(2));
    params.D = static_cast<int>(q.size(3));
    params.scale = 1.0f / sqrt(static_cast<float>(params.D));

    dim3 grid(params.B, params.H, flash_config::NUM_SPLITS);  // 一个三维grid，表示Block数量为：B * H * num_splits
    constexpr int threads = 128;                // 每个block的线程数

    // 获取 PyTorch 当前 CUDA stream
    cudaStream_t stream =
    c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    flash_fwd_kernel<<<grid, threads, 0, stream>>>(params);

    auto status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));

    return out;
}


// 4. 暴露给Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_forward, "Flash Attention Forward");
}