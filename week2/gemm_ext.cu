#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cublasLt.h>
#include <cuda_runtime.h>


torch::Tensor gemm_forward(
    torch::Tensor a, 
    torch::Tensor b
){
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA");

    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");

    TORCH_CHECK(a.scalar_type() == torch::kFloat16,
                "a must be FP16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16,
                "b must be FP16");

    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    auto output = torch::empty(
        {M, N},
        torch::TensorOptions()
        .device(a.device())
        .dtype(torch::kFloat32)
    );

    // 添加cuBLASLt 调用
    cublasLtHandle_t handle;   
    cublasLtCreate(&handle);

    cublasLtMatmulDesc_t operation_desc;
    cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // 创建矩阵描述符
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatrixLayout_t d_desc;

    cublasLtMatrixLayoutCreate(
        &a_desc,
        CUDA_R_16F,
        M, 
        K,
        K
    );

    cublasLtMatrixLayoutCreate(
        &b_desc,
        CUDA_R_16F,
        K, 
        N,
        N
    );

    cublasLtMatrixLayoutCreate(
        &c_desc,
        CUDA_R_32F,
        M, 
        N,
        N
    );

    cublasLtMatrixLayoutCreate(
        &d_desc,
        CUDA_R_32F,
        M, 
        N,
        N
    );

    // 设置4个矩阵描述符为行主序
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    cublasLtMatrixLayoutSetAttribute(
        a_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    );

    cublasLtMatrixLayoutSetAttribute(
        b_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    );

    cublasLtMatrixLayoutSetAttribute(
        c_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    );

    cublasLtMatrixLayoutSetAttribute(
        d_desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    );

    // 矩阵乘法
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.device().index()).stream();
    // cublasLtMatmul: D = alpha * op(A) * op(B) + beta * C
    cublasStatus_t status = cublasLtMatmul(
        handle,
        operation_desc,
        &alpha,
        a.data_ptr<at::Half>(),
        a_desc,
        b.data_ptr<at::Half>(),
        b_desc,
        &beta,
        output.data_ptr<float>(),
        c_desc,
        output.data_ptr<float>(),
        d_desc,
        nullptr,
        nullptr,
        0,
        stream
    );

    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "cublasLtMatmul failed, status = ",
        static_cast<int>(status)
    );

    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(d_desc);
    cublasLtMatmulDescDestroy(operation_desc);
    cublasLtDestroy(handle);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward);
}