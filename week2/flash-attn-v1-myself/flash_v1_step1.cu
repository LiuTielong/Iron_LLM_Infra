/*
 * FlashAttention v1 - Step 1: 官方风格的调度与分块加载
 *
 * 本文件故意只实现最基础的一层：
 *
 *   Global Memory 中的 Q/K/V
 *       -> Shared Memory 中的 Q/K/V tile
 *       -> 把 Q tile 原样写回输出，验证 Q tile 的访问
 *
 * 这里还没有实现：
 *   - QK^T
 *   - causal mask
 *   - Online Softmax
 *   - P @ V
 *
 * 因此 forward() 当前返回的是 Q 的拷贝，不是 Attention 输出。
 * 这个文件的重点是先把官方 FlashAttention v1 的循环顺序搭正确：
 *
 *   for (kv_tile : all KV tiles) {       // 外层：KV tile
 *       load K/V tile once;
 *
 *       for (q_tile : this block owns) { // 内层：Q tile
 *           load Q tile;
 *           // 后续将在这里加入 QK^T、Softmax、P@V
 *       }
 *   }
 *
 * 输入布局固定为连续的 [B, S, H, D]：
 *   B: batch size
 *   S: sequence length
 *   H: attention heads
 *   D: head dimension，当前固定为 128
 *
 * 本阶段使用 FP16 或 BF16 输入；数据搬运本身不进行类型转换。
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <climits>

namespace flash_config {

// 一个 Query tile 包含 Br 行，每一行长度为 D。
constexpr int HEAD_DIM = 128;
constexpr int QUERY_TILE = 16;

// 一个 KV tile 包含 Bc 行。K 和 V 都使用这个 tile 大小。
constexpr int KV_TILE = 128;

// 当前使用 128 个线程，即 4 个 warp。
constexpr int THREADS = 128;

// grid.z 表示 Query tile 的分片数。
// NUM_Q_SPLITS=1 时，一个 block 顺序处理所有 Q tile；
// 改成 2、4 等值时，多个 block 会在 Q 方向分担工作。
constexpr int NUM_Q_SPLITS = 1;

}  // namespace flash_config

/*
 * 传给 GPU kernel 的参数。
 *
 * 这是一个普通 C++ struct。kernel 接收它的一个副本；
 * 其中的 q/k/v/o/debug 指针本身指向 GPU memory。
 */
struct FlashFwdParams {
    const void* q;       // Q 的 GPU 地址，元素类型是 FP16 或 BF16
    const void* k;       // K 的 GPU 地址
    const void* v;       // V 的 GPU 地址
    void* o;              // 普通 forward 的输出地址；debug_load 时可以为空
    void* debug;          // 可选的 debug 输出地址；普通 forward 时为空

    int B;
    int S;
    int H;
    int D;
    int num_kv_tiles;
    float scale;         // 下一阶段 QK^T 会使用；本阶段暂未使用
};

/*
 * 第一步 kernel。
 *
 * template<typename scalar_t>
 *   编译器会根据 host 端的分支生成两个版本：
 *       flash_v1_step1_kernel<__half>
 *       flash_v1_step1_kernel<__nv_bfloat16>
 *
 * __global__
 *   表示这个函数由 CPU 通过 <<<grid, block, smem, stream>>> 启动，
 *   但函数体由 GPU 上的所有线程执行。
 *
 * 本阶段的 block 映射：
 *   blockIdx.x -> batch
 *   blockIdx.y -> attention head
 *   blockIdx.z -> Query tile 的分片编号
 *
 * q_tile 从 blockIdx.z 开始，每次增加 gridDim.z。
 * 例如 num_q_tiles=8、gridDim.z=2 时：
 *   z=0 的 block 处理 q_tile 0,2,4,6；
 *   z=1 的 block 处理 q_tile 1,3,5,7。
 */
template <typename scalar_t>
__global__ void flash_v1_step1_kernel(FlashFwdParams params) {
    // constexpr 表示编译期常量；编译器可以据此展开循环和计算偏移。
    constexpr int Br = flash_config::QUERY_TILE;
    constexpr int Bc = flash_config::KV_TILE;
    constexpr int D = flash_config::HEAD_DIM;

    // threadIdx.x 是当前线程在线程块内的编号，范围为 0～127。
    const int tid = threadIdx.x;

    // 同一个 block 的线程共享 blockIdx；不同 block 的 blockIdx 不同。
    const int batch = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int q_split = static_cast<int>(blockIdx.z);
    const int num_q_splits = static_cast<int>(gridDim.z);
    // void* 不携带元素类型。这里把它恢复为实际的数据类型指针。
    // static_cast 只改变指针的解释方式，不会复制或转换数据。
    const scalar_t* q_ptr = static_cast<const scalar_t*>(params.q);
    const scalar_t* k_ptr = static_cast<const scalar_t*>(params.k);
    const scalar_t* v_ptr = static_cast<const scalar_t*>(params.v);
    scalar_t* o_ptr = static_cast<scalar_t*>(params.o);
    scalar_t* debug_ptr = static_cast<scalar_t*>(params.debug);
    /*
     * 动态 shared memory：每一个 block 拥有一份，block 内所有线程共享。
     *
     * extern __shared__ 的数组长度不在源代码中写死，
     * 而是在 kernel 启动时由第三个 <<<...>>> 参数传入字节数。
     *
     * 这一块连续内存被切成三个区域：
     *
     *   q_s: [Br, D] = [16, 128]
     *   k_s: [Bc, D] = [128, 128]
     *   v_s: [Bc, D] = [128, 128]
     *
     * unsigned char 每个元素占 1 字节，因此先把它当作原始字节地址，
     * 再用 reinterpret_cast 解释成 scalar_t*。
     */
    extern __shared__ __align__(16) unsigned char smem_raw[];

    scalar_t* q_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* k_s = q_s + Br * D;
    scalar_t* v_s = k_s + Bc * D;

    /*
     * 连续 [B, S, H, D] tensor 的地址公式：
     *
     *   offset = (((batch * S + seq_pos) * H + head) * D + col)
     *
     * 固定 batch 和 head 后，可以写成：
     *
     *   offset = head_base + seq_pos * row_stride + col
     *
     * row_stride = H * D，表示序列位置增加 1 时要跨过的元素数量。
     * 使用 int64_t 是为了让较大的矩阵尺寸下，地址乘法不容易发生
     * 32 位整数溢出。
     */
    const int64_t row_stride =
        static_cast<int64_t>(params.H) * D;
    const int64_t head_base =
        static_cast<int64_t>(batch) * params.S * row_stride
        + static_cast<int64_t>(head) * D;

    const int num_q_tiles =
        (params.S + Br - 1) / Br;

    /*
     * 官方 v1 风格的外层循环：依次处理 KV tile。
     *
     * 对 S=512、Bc=128：
     *   kv_tile = 0, 1, 2, 3
     *
     * 当前 KV tile 被加载到 k_s/v_s 后，会在内层被多个 Q tile 复用。
     * 这正是当前版本与之前 q_tile 外层实现的关键区别。
     */
    for (int kv_tile = 0;
         kv_tile < params.num_kv_tiles;
         ++kv_tile) {

        const int kv_start = kv_tile * Bc;

        /*
         * 所有线程协作加载 K tile 和 V tile。
         *
         * Bc * D = 128 * 128 = 16384 个元素，
         * 128 个线程通过步长 blockDim.x 共同完成搬运。
         *
         * 线程 tid 负责：
         *   idx = tid, tid + 128, tid + 256, ...
         *
         * 因而每个线程为 K 搬运 128 个元素，
         * 为 V 再搬运 128 个元素。
         */
        for (int idx = tid;
             idx < Bc * D;
             idx += blockDim.x) {

            // 一维 tile 下标转换成 tile 内的行号和列号。
            const int row = idx / D;
            const int col = idx % D;
            const int seq_pos = kv_start + row;

            if (seq_pos < params.S) {
                const int64_t offset =
                    head_base
                    + static_cast<int64_t>(seq_pos) * row_stride
                    + col;

                // Global Memory -> Shared Memory。
                k_s[idx] = k_ptr[offset];
                v_s[idx] = v_ptr[offset];
            } else {
                // 最后一个 KV tile 可能不完整，越界位置填 0。
                // 下一阶段做 Softmax 时仍必须额外屏蔽这些位置。
                k_s[idx] = scalar_t(0);
                v_s[idx] = scalar_t(0);
            }
        }

        /*
         * block 内同步：
         * 保证所有线程都完成 K/V 的写入后，任何线程才能读取 k_s/v_s。
        */
        __syncthreads();

        /*
         * debug_load() 会传入 debug_ptr，用于验证 K/V tile 的内容。
         * 普通 forward() 不传 debug_ptr，因此不会执行这一段。
         *
         * debug tensor 的布局为：
         *   [B, H, num_q_splits, num_kv_tiles, 2, Bc, D]
         *
         * 第 2 维的 2 表示：
         *   0 -> K tile
         *   1 -> V tile
         */
        if (debug_ptr != nullptr) {
            const int64_t tile_elements =
                static_cast<int64_t>(2) * Bc * D;

            const int64_t block_debug_base =
                ((static_cast<int64_t>(batch) * params.H + head)
                 * num_q_splits + q_split)
                * params.num_kv_tiles * tile_elements
                + static_cast<int64_t>(kv_tile) * tile_elements;

            for (int idx = tid;
                 idx < Bc * D;
                 idx += blockDim.x) {
                debug_ptr[block_debug_base + idx] = k_s[idx];
                debug_ptr[block_debug_base + Bc * D + idx] = v_s[idx];
            }
        }

        // 确保 debug 写入完成后，所有线程再进入 Q tile 循环。
        __syncthreads();

        /*
         * 官方 v1 风格的内层循环：处理当前 block 所负责的 Q tiles。
         *
         * q_tile 从 q_split 开始，每次跳过 num_q_splits。
         * 这样不同 z 方向的 block 可以分担 Q tile。
         */
        for (int q_tile = q_split;
             q_tile < num_q_tiles;
             q_tile += num_q_splits) {

            const int q_start = q_tile * Br;

            /*
             * 加载当前 Q tile。
             *
             * 与 K/V 加载相同，128 个线程协作搬运 Br*D 个元素。
             * 由于 Q tile 比 KV tile 小，每个线程平均搬运 16 个元素。
             */
            for (int idx = tid;
                 idx < Br * D;
                 idx += blockDim.x) {

                const int row = idx / D;
                const int col = idx % D;
                const int seq_pos = q_start + row;

                if (seq_pos < params.S) {
                    const int64_t offset =
                        head_base
                        + static_cast<int64_t>(seq_pos) * row_stride
                        + col;
                    q_s[idx] = q_ptr[offset];
                } else {
                    // 最后一个 Query tile 的越界行填 0。
                    q_s[idx] = scalar_t(0);
                }
            }

            // 确保整个 Q tile 加载完成。
            __syncthreads();

            /*
             * 当前阶段的验证操作：把 Q tile 写回 O。
             *
             * 这不是 Attention 输出，只用于检查：
             *   1. q_tile 的计算是否正确；
             *   2. [B,S,H,D] 的地址公式是否正确；
         *   3. blockIdx.z 的分片是否导致重复或遗漏。
             *
             * 因为官方调度是 KV tile 外层，所以同一个 Q tile 会被
             * 每个 KV tile 重新写一次 O；每次写入的内容都相同，
             * 最终结果仍应等于 Q。
             */
            if (o_ptr != nullptr) {
                for (int idx = tid;
                     idx < Br * D;
                     idx += blockDim.x) {

                    const int row = idx / D;
                    const int col = idx % D;
                    const int seq_pos = q_start + row;

                    if (seq_pos < params.S) {
                        const int64_t offset =
                            head_base
                            + static_cast<int64_t>(seq_pos) * row_stride
                            + col;
                        o_ptr[offset] = q_s[idx];
                    }
                }
            }

            /*
             * 下一次 q_tile 会覆盖 q_s。
             * 所以必须保证当前 Q tile 的所有读取和写回都完成后，
             * 才允许任何线程开始写入下一个 Q tile。
             */
            __syncthreads();
        }

        /*
         * 下一轮 kv_tile 会覆盖 k_s/v_s。
         * 这里的同步保证当前 KV tile 已经不会再被当前 block 使用。
         *
         * 当某个 block 没有分配到有效 q_tile 时，虽然它不会进入内层
         * 循环，但所有线程仍然会到达这里，因此同步条件是一致的。
         */
        __syncthreads();
    }
}

/*
 * 启动指定 scalar_t 版本的 kernel。
 * 把 FP16/BF16 两个版本的公共启动逻辑放在模板函数里，
 * 避免 CPU 接口中重复两大段代码。
 */
template <typename scalar_t>
void launch_step1_kernel(
    const FlashFwdParams& params,
    dim3 grid,
    int threads,
    size_t smem_bytes,
    cudaStream_t stream
) {
    // 当前 shared memory 约为 68 KiB，超过许多设备的默认上限。
    cudaError_t status = cudaFuncSetAttribute(
        flash_v1_step1_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    TORCH_CHECK(
        status == cudaSuccess,
        "failed to set dynamic shared memory size: ",
        cudaGetErrorString(status));

    /*
     * CUDA kernel 启动语法：
     *
     *   kernel<<<grid, threads, smem_bytes, stream>>>(params);
     *
     * 四个 <<< >>> 参数依次是：
     *   1. grid 的维度
     *   2. 每个 block 的线程数
     *   3. 每个 block 的动态 shared memory 字节数
     *   4. 使用的 CUDA stream
     */
    flash_v1_step1_kernel<scalar_t>
        <<<grid, threads, smem_bytes, stream>>>(params);

    status = cudaGetLastError();
    TORCH_CHECK(
        status == cudaSuccess,
        "flash_v1_step1_kernel launch failed: ",
        cudaGetErrorString(status));
}

/*
 * 检查输入，并计算本阶段需要的 launch 参数。
 * 这个函数不会启动 kernel，只是把公共参数填入 params。
 */
FlashFwdParams make_params(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK(
        q.is_cuda() && k.is_cuda() && v.is_cuda(),
        "q, k, v must be CUDA tensors");
    TORCH_CHECK(
        q.device() == k.device() && q.device() == v.device(),
        "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.dim() == 4, "expected layout [B, S, H, D]");
    TORCH_CHECK(
        q.sizes() == k.sizes() && q.sizes() == v.sizes(),
        "q, k, v must have the same shape");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
            q.scalar_type() == torch::kBFloat16,
        "q, k, v must be FP16 or BF16");
    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() &&
            q.scalar_type() == v.scalar_type(),
        "q, k, v must have the same dtype");
    TORCH_CHECK(
        q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "q, k, v must be contiguous");
    TORCH_CHECK(
        q.size(3) == flash_config::HEAD_DIM,
        "head_dim must be 128");
    TORCH_CHECK(q.numel() > 0, "input must not be empty");

    TORCH_CHECK(q.size(0) <= INT_MAX, "B is too large");
    TORCH_CHECK(q.size(1) <= INT_MAX, "S is too large");
    TORCH_CHECK(q.size(2) <= INT_MAX, "H is too large");

    FlashFwdParams params{};
    params.q = q.data_ptr();
    params.k = k.data_ptr();
    params.v = v.data_ptr();
    params.o = nullptr;
    params.debug = nullptr;

    params.B = static_cast<int>(q.size(0));
    params.S = static_cast<int>(q.size(1));
    params.H = static_cast<int>(q.size(2));
    params.D = static_cast<int>(q.size(3));
    params.num_kv_tiles =
        (params.S + flash_config::KV_TILE - 1)
        / flash_config::KV_TILE;
    params.scale =
        1.0f / std::sqrt(static_cast<float>(params.D));

    return params;
}

/*
 * 启动公共逻辑。
 * q 用于确定 dtype、device 和 shared memory 大小。
 */
void launch_from_params(
    FlashFwdParams params,
    torch::Tensor q
) {
    c10::cuda::CUDAGuard device_guard(q.device());

    // 每个 block 负责一个 batch/head；z 维表示 Q tile 分片。
    dim3 grid(
        params.B,
        params.H,
        flash_config::NUM_Q_SPLITS);
    constexpr int threads = flash_config::THREADS;

    // Q + K + V 三个 shared-memory tile 的总元素数量。
    constexpr int smem_elements =
        (flash_config::QUERY_TILE
         + 2 * flash_config::KV_TILE)
        * flash_config::HEAD_DIM;

    const size_t smem_bytes =
        static_cast<size_t>(smem_elements) * q.element_size();

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    if (q.scalar_type() == torch::kFloat16) {
        launch_step1_kernel<__half>(
            params, grid, threads, smem_bytes, stream);
    } else {
        launch_step1_kernel<__nv_bfloat16>(
            params, grid, threads, smem_bytes, stream);
    }
}

/*
 * 普通 forward 接口。
 *
 * 当前返回 Q 的原样拷贝，仅用于验证 Q tile 的调度和地址计算。
 */
torch::Tensor flash_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    FlashFwdParams params = make_params(q, k, v);

    auto out = torch::empty_like(q);
    params.o = out.data_ptr();

    launch_from_params(params, q);
    return out;
}

/*
 * 可选的 debug 接口，用于验证 K/V tile 的加载结果。
 * 返回布局：
 *   [B, H, NUM_Q_SPLITS, num_kv_tiles, 2, Bc, D]
 *
 * 最后一维组中的 2 表示：
 *   0 -> K tile
 *   1 -> V tile
 *
 * 最后一个不完整 KV tile 的越界行应当是 0。
 */
torch::Tensor flash_debug_load(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    FlashFwdParams params = make_params(q, k, v);

    const int64_t num_kv_tiles = params.num_kv_tiles;
    auto debug = torch::empty(
        {q.size(0),
         q.size(2),
         flash_config::NUM_Q_SPLITS,
         num_kv_tiles,
         2,
         flash_config::KV_TILE,
         flash_config::HEAD_DIM},
        q.options());

    params.debug = debug.data_ptr();
    // debug_load 不需要普通输出，因此 params.o 保持 nullptr。
    launch_from_params(params, q);
    return debug;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &flash_forward,
        "FlashAttention v1 step 1: official-order tile loading");
    m.def(
        "debug_load",
        &flash_debug_load,
        "Debug K/V tile loading for FlashAttention v1 step 1");
}
