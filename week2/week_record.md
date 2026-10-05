# 第一步：独立实现稳定 Softmax
## 主要方法
每一个Block处理一行数据。我们这里将Block分成256个线程。对于1024这个维度来说，每个线程处理4个数据：先规约找最大值；再规约求和；最后输出4个y。


## 实验结果
random max error: 3.73e-09
large values max error: 3.73e-09

shape=(128, 1024), dtype=torch.float32
implementation         avg_ms       GB/s
custom               0.009984     105.03
torch.softmax        0.011172      93.86
speedup = 1.12x

## 进行CUB库调用
CUB（CUDA Unbound）是一个专为 CUDA 内核编程设计的 C++ 模板库，它提供了高性能、可复用的并行计算组件，让你无需从零实现复杂的并行算法，就能写出“光速级”的 CUDA 内核。


# 第二步：建立标准Attention 推理基线


# 第三步：调用基础库并准备 Tensor Core 矩阵乘法
## 步骤
1. 先支持两个矩阵的乘法
C = A @ B. A, B 为FP16，C为FP32. 
2. 先创建接口函数骨架：输入的tensor a, b. 需要检查它们是否在gpu上，它们的维度是否正确。另外创建一个输出输出tensor c，设置大小。然后，暴露给Python调用。
3. 写一个python 程序，调用这个接口函数。
4. 创建 cuBLASLt handle，准备调用这个库。
5. 添加矩阵乘法描述符：cublasLtMatmulDesc_t。
6. 创建4个矩阵布局描述符：cublasLtMatrixLayout_t。表示矩阵A, B, C, D。
7. 调用矩阵乘法：cublasLtMatmul()
8. 在Python中准备数据并进行调用、检查功能是否正确。

实验设置：
矩阵大小都是1024*1024. 

实验结果：
custom               0.056652     148.07
torch.matmul         0.027707     302.76
speedup: 0.49x
可以看到，比torch.matmul 要慢不少啊。

# 第四步：实现 FlashAttention v1 核心算法并接入 Prefill
## 步骤
### 1.参数范围
输入布局：Q/K/V [B, S, H, D]
输出布局：O     [B, S, H, D]
D = 128
FP16/BF16 输入
FP32 Online Softmax
causal Prefill
只实现 forward
### 2. 建立Extension骨架
还是写一个.cu 程序，然后创建一个.py 文件，调用这个.cu 文件。

对于.cu 程序，分成四块：
```
// 1. 从CPU传给GPU kernel的参数
// 2. GPU kernel函数
// 3. CPU 接口：检查 Tensor、准备参数、启动 kernel
// 4. 暴露给Python
```

编译运行的命令：
```
conda activate longspec_new
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

cd /home/liutielong/Files_2026/LLM_Infra/week2/flash-attn-v1-myself
CUDA_VISIBLE_DEVICES=6 python test_flash_v1.py
```

### 3. 确定kernel配置
每个block处理一个batch、一个head、一个query tile。
先将S固定为512， B固定为1. 
Query tile: 16行。
Key/Value tile: 128一行。
head_dim: 128。
线程数:128. 

所以第一个GEMM：
```
Q_tile [16, 128] @ K_tile^T [128, 128]
→ scores [16, 128]
```

第二个GEMM：
```
P_tile [16, 128] @ V_tile [128, 128]
→ O_tile [16, 128]
```

我们将num_splits 固定为1. 这样，每个block处理一个head（因为batch size 也是1）。
然后，由于KV tile = 128, 所以每个block在外层循环4次。另外，Q tile = 16, 所以每个block还会在内层循环32次。

### 4. 实现Q/K/V的分块加载
Q_tile: [16, 128]
K_tile: [128, 128]
V_tile: [128, 128]
### 5. 接入Tensor Core MMA
使用官方 v1 的 CUTLASS MMA 和 mma.sync 封装，计算：Q_tile @ K_tile^T。
输出分数只保存在寄存器或shared memory 中，不写入 global memory。
### 6. 实现causal mask 和 Online Softmax
### 7. 接入第二个Tensor Core MMA: 计算 P @ V_tile
### 8. 完成正确性测试
### 9. 进行性能测试
### 10. 最后接入 LLaMA-2