/*
这一步主要进行数据的分块搬运
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
constexpr int HEAD_DIM = 128;
constexpr int QUERY_TILE = 16;
constexpr int KV_TILE = 128;
constexpr int THREADS = 128;
constexpr int NUM_Q_SPLITS = 1;
}

struct FlashFwdParams {
    const void* q;       
    const void* k;       
    const void* v;       
    void* o;             
    void* debug;         

    int B;
    int S;
    int H;
    int D;
    int num_kv_tiles;
    float scale;         
};

template <typename scalar_t>
__global__ void flash_v1_step1_kernel(FlashFwdParams params) {
    // 定义常量
    constexpr int Br = flash_config::QUERY_TILE;
    constexpr int Bc = flash_config::KV_TILE;
    constexpr int D = flash_config::HEAD_DIM;

    const int tid = threadIdx.x;