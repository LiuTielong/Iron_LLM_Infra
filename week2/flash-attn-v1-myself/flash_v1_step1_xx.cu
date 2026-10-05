/*
文件功能：实现“分块加载”：
    1. Global memory 中的Q/K/V.
    2. 按tile（q_s, k_s, v_s) 加载到每个block 的shared memory 中。
    3. 暂时把Q tile 写回输出。
*/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace flash_config {
    constexpr int HEAD_DIM = 128;
    constexpr int NUM_HEADS = 32;
    constexpr int QUERY_TILE = 16;
    constexpr int KV_TILE = 128;
    constexpr int THREADS = 128;
}

// CPU 端打包传给GPU kernel的参数。
// 里面的指针指向 GPU memory。
struct FlashFwdParams {
    const void* q;
    const void* k;
    const void* v;
    void* o;

    int B;
    int S;
    int H;
    int D;
    float scale;
};

template <typename scalar_t>
__global__ void flash_fwd_kernel(FlashFwdParams params){
    // 1. 定义常量
    constexpr int Br = flash_config::QUERY_TILE;
    constexpr int Bc = flash_config::KV_TILE;
    constexpr int H = flash_config::NUM_HEADS;
    constexpr int D = flash_config::HEAD_DIM;

    const int tid = threadIdx.x;
    const int64_t batch = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int q_tile = blockIdx.z;

    const int q_start = q_tile * Br;

    const scalar_t* q_ptr = static_cast<const scalar_t*>(params.q);
    const scalar_t* k_ptr = static_cast<const scalar_t*>(params.k);
    const scalar_t* v_ptr = static_cast<const scalar_t*>(params.v);
    scalar_t* o_ptr = static_cast<scalar_t*>(params.o);

    // 2. 定义 shared memory
    // 并分配给 Q/K/V tile
    extern __shared__ __align__(16) unsigned char smem_raw[];
    scalar_t* q_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* k_s = q_s + Br * D;
    scalar_t* v_s = k_s + Bc * D;

    const int64_t row_stride = H * D;
    const int64_t head_base = batch * params.S * H * D + head * D;

    // 第一阶段：线程协作加载Q tile. 
    // 每个线程加载16个Q 元素
    for (int idx = tid; idx < Br * D; idx += blockDim.x){
        const int row = idx / D;
        const int col = idx % D;
        const int seq_pos = q_start + row;
        if (seq_pos < params.S){
            const int64_t offset =
                head_base + seq_pos * row_stride + col;
            q_s[idx] = q_ptr[offset];
        } else {
            q_s[idx] = scalar_t(0);
        }
    }
    __syncthreads();

    // 第二阶段：循环加载所有KV tiles。
    // 每个线程加载128个K/V 元素
    for (int kv_start=0; kv_start < params.S; kv_start += Bc){
        for (int idx = tid; idx < Bc * D; idx += BlockDim.x){
            const int row = idx / D;
            const int col = idx % D;
            const int seq_pos = kv_start + row;

            if (seq_pos < params.S){
                const int64_t offset = 
                    head_base + seq_pos * row_stride + col;

                k_s[idx] = k_ptr[offset];
                v_s[idx] = v_ptr[offset];
            } else {
                k_s[idx] = scalar_t(0);
                v_s[idx] = scalar_t(0);
            }
        }
    }
    __syncthreads();

    // 第三阶段：执行计算
    // 当前暂未实现

    // 第四阶段：写回输出
    // 当前就将q_s 的 Q_tile 写回O，作为验证。
    for (int idx = tid; idx < Br * D; idx += blockDim.x) {
        const int row = idx / D;
        const int col = idx % D;
        const int seq_pos = q_start + row;

        if (seq_pos < params.S) {
            const int64_t offset = 
                head_base + seq_pos * row_stride + col;
            o_ptr = q_s[idx];
        }
    }
}

// CPU 端的Extension 接口
torch::Tensor flash_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.dim() == 4, "expected layout [B, S, H, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, v must have the same shape");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 ||
                    q.scalar_type() == torch::kBFloat16,
                "q, k, v must be FP16 or BF16");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                    q.scalar_type() == v.scalar_type(),
                "q, k, v must have the same dtype");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, v must be contiguous");
    TORCH_CHECK(q.size(3) == flash_config::HEAD_DIM,
                "head_dim must be 128");
    TORCH_CHECK(q.numel() > 0, "input must not be empty");

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
    params.scale = 1.0f / std::sqrt(static_cast<float>(params.D));

    // 让每个query tile 拥有一个block
    const int num_q_tiles = 
        (params.S + flash_config::QUERY_TILE - 1) / flash_config::QUERY_TILE;
    dim3 grid(params.B, params.H, num_q_tiles);
    constexpr int threads = flash_config::THREADS;

    // 每个block的shared memory大小
    constexpr int smem_elements = 
        (flash_config::QUERY_TILE + 2 * flash_config::KV_TILE) * flash_config::HEAD_DIM;
    const size_t smem_bytes = smem_elements * q.element_size();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    // 所需空间超过默认的 48 KiB，需要用 cudaFuncSetAttribute 显式允许 68 KiB。
    // 此处设置某个 kernel 的动态 shared memory 上限；实际空间由 launch 第三参指定。
    cudaError_t status;
    if (q.scalar_type() == torch::kFloat16) {
        status = cudaFuncSetAttribute(
            flash_fwd_kernel<__half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        TORCH_CHECK(status == cudaSuccess,
                    "failed to set FP16 shared memory size: ",
                    cudaGetErrorString(status));

        // 启动kernel
        flash_fwd_kernel<__half><<<grid, threads, smem_bytes, stream>>>(params);
    } else {
        status = cudaFuncSetAttribute(
            flash_fwd_kernel<__nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        TORCH_CHECK(status == cudaSuccess,
                    "failed to set BF16 shared memory size: ",
                    cudaGetErrorString(status));
        
        flash_fwd_kernel<__nv_bfloat16><<<grid, threads, smem_bytes, stream>>>(params);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_forward, "Flash Attention v1 step1 forward"); 
}