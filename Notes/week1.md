# Week 1：CUDA Kernel 与性能分析

## 1. 本周目标

本周对应 `Infra.md` 第 4.1 节，是第一阶段的 CUDA 基础周。目标不是简单地“写出几个 Kernel”，而是完成从参考实现、CUDA 实现、正确性验证、性能测量到 Profiling 分析的第一个闭环。

本周结束时，应能够：

- 解释 Thread、Warp、Block、Grid、SM、SIMT 以及 Kernel Launch 的基本关系；
- 解释 Global Memory、L2、L1/Texture Cache、Shared Memory 和 Register 的作用；
- 识别合并访存、Warp Divergence、同步开销、寄存器压力和 Bank Conflict 等常见问题；
- 正确使用 CUDA Event 测量异步 Kernel；
- 独立实现并验证 RMSNorm 和 Softmax 的 PyTorch Reference、Naive CUDA、Optimized CUDA 版本；
- 用性能数据判断 Kernel 更接近 Memory-bound、Compute-bound 还是 Launch-bound；
- 使用 Nsight Systems 或 Nsight Compute 对至少一个 Kernel 找出真实瓶颈，并解释一次有效优化。

如果本周时间不足，优先保证 **RMSNorm 的完整优化与 Profiling**；Softmax 至少需要完成正确的 CUDA 版本、正确性测试和基础 Benchmark，不能省略测试。

## 2. 当前实验环境

GPU 实验应登录 GPU 节点执行，`master` 节点只用于联网、下载和准备环境。

当前已确认的环境：

- GPU：NVIDIA A100 PCIe 40GB，Compute Capability 8.0；
- GPU 节点驱动：570.133.20；
- CUDA Toolkit：12.8，路径为 `/usr/local/cuda`；
- PyTorch：2.5.1+cu124；
- Triton：3.1.0；
- Nsight Systems：2024.6.2；
- Nsight Compute：2025.1.0；
- PyTorch 内置 NCCL：2.21.5。

开始实验前，在 GPU 节点执行：

```bash
ssh gpu01
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
```

目前 `gpu01` 的 GPU 5、6，以及 `gpu02`/`gpu03` 的部分高编号 GPU 相对空闲；具体使用哪张卡必须在启动前重新检查：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=5 python3 your_script.py
```

不要占用已有任务的 GPU，也不要使用 `gpu04` 上几乎已经占满的 GPU。Benchmark 记录中必须写明节点、物理 GPU 编号、驱动、CUDA、PyTorch 和 dtype。

## 3. 本周时间安排

总投入约 15～20 小时。下面的天数可以按照实际时间平移，但建议保持先基础、后 Kernel、再 Profiling 的顺序。

| 学习单元 | 时间 | 主要内容 | 当天产出 |
|---|---:|---|---|
| Day 1：环境与执行模型 | 2～3 h | CUDA 执行模型、A100 信息、Vector Add | 环境检查记录、Vector Add |
| Day 2：访存、同步与 Reduction | 2～3 h | 合并访存、Warp、Shared Memory、Warp Shuffle、CUDA Event | Naive/Optimized Reduction |
| Day 3：RMSNorm 基础实现 | 3 h | 公式、PyTorch Reference、Naive CUDA、误差测试 | 可运行 RMSNorm Naive |
| Day 4：RMSNorm 优化 | 3～4 h | 行级并行、向量化 Load/Store、Warp Reduction、FP32 Accumulation | RMSNorm Optimized、初版性能表 |
| Day 5：Softmax 实现 | 3～4 h | 数值稳定 Softmax、Max/Sum Reduction、Naive/Optimized CUDA | Softmax 两个 CUDA 版本 |
| Day 6：Benchmark 与 Nsight | 2～3 h | 多 Shape 测试、CUDA Event、Nsight、瓶颈判断 | 完整性能表、Profiling 数据 |
| Day 7：整理与面试复盘 | 1～2 h | 报告、图表、设计取舍、口头讲解 | 2～3 页 Profiling 笔记、验收清单 |

## 4. 分日任务

### Day 1：环境检查与 CUDA 执行模型

学习：

- Thread、Warp、Block、Grid、SM 与 SIMT；
- Kernel Launch、异步执行和 `cudaDeviceSynchronize()`；
- A100 的 Compute Capability 8.0、Warp 大小和基本硬件限制；
- Host、Device、Kernel、Stream 的基本关系。

实践：

1. 在一个空闲 GPU 上记录 `nvidia-smi -L`、`nvidia-smi` 和 `nvidia-smi topo -m`；
2. 编译并运行 Vector Add；
3. 分别比较未同步计时、CPU wall-clock 计时和 CUDA Event 计时；
4. 写一个最小 PyTorch CUDA Smoke Test，确认输入、输出和同步都在预期设备上。

需要回答：

- 为什么 Kernel Launch 返回时 GPU 计算可能还没有完成？
- 为什么只用 Python `time.time()` 很容易低估或误判 Kernel 时间？
- Block 大小改变时，哪些资源会变化？

### Day 2：访存、同步与 Reduction

学习：

- Global Memory 合并访存和数据对齐；
- Register、Shared Memory、L1/L2 Cache 的基本用途；
- `__syncthreads()` 的作用与限制；
- Shared Memory Reduction 与 Warp Shuffle Reduction；
- Warp Divergence 和不规则分支。

实践：

1. 实现 Sum Reduction 的 Naive 版本；
2. 实现使用 Shared Memory 的版本；
3. 实现使用 Warp Shuffle 的版本；
4. 对不同输入规模进行正确性和延迟对比；
5. 观察 Block 大小、同步次数和访存模式对性能的影响。

Reduction 只作为训练和环境验证，不作为最终作品集的主要 Kernel。它应帮助后续理解 RMSNorm 和 Softmax 中的行内归约。

### Day 3：RMSNorm 基础实现

对输入最后一维做 RMSNorm。参考公式为：

```text
rms(x) = sqrt(mean(x^2) + eps)
y = x / rms(x) * weight
```

实践：

1. 用 PyTorch 写 Reference，使用 FP32 累加平方和；
2. 将输入视为二维矩阵 `[rows, hidden_size]`，每一行独立归一化；
3. 用 CUDA C++ 实现 Naive Kernel；
4. 使用 PyTorch Extension 或等价方式从 Python 调用 CUDA Kernel；
5. 支持至少 FP32 和 FP16，尽可能补充 BF16；
6. 对输出执行 `torch.testing.assert_close`，记录最大绝对误差和最大相对误差。

第一版可以采用“一行对应一个 Block”的简单设计。重点是先确保：索引正确、边界正确、累加精度正确、Kernel 调用后正确同步。

### Day 4：RMSNorm 优化

在 Naive 版本正确的基础上，每次只引入一种可解释的优化，并单独测量收益：

- 使用 Warp Shuffle 完成平方和 Reduction，减少 Shared Memory 和同步；
- 对满足对齐条件的输入尝试向量化 Load/Store，例如 `float4` 或适当的 half 向量类型；
- 使用 FP32 累加，避免 FP16/BF16 Reduction 的明显数值误差；
- 减少中间结果写回，尽量在同一个 Kernel 内完成归一化和缩放；
- 比较不同 Block Size 和每个线程处理元素数量。

优化时必须同时记录：

- 优化前后延迟；
- 读取/写入字节数和有效带宽；
- 寄存器数量、Occupancy 或 Warp Stall 的变化；
- 数值误差是否变化；
- 优化在哪些 Shape 上有效，在哪些 Shape 上无效。

不要把“Occupancy 更高”直接等同于“性能更好”。如果 Kernel 受 HBM 带宽或指令依赖限制，提高 Occupancy 可能没有收益，甚至会因寄存器减少、局部性变差而变慢。

### Day 5：Softmax 实现

对最后一维实现数值稳定的 Softmax：

```text
m = max(x)
s = sum(exp(x - m))
y = exp(x - m) / s
```

实践：

1. 写 PyTorch Reference，并明确 `dim=-1` 的语义；
2. 实现 Naive CUDA 版本；
3. 实现使用 Warp/Block Reduction 的 Optimized CUDA 版本；
4. Max 和 Sum 阶段使用 FP32 累加或中间精度；
5. 覆盖普通值、较大正数、较大负数以及包含重复最大值的输入；
6. 与 `torch.softmax` 做误差对比，并检查是否出现 NaN/Inf。

Softmax 的重点是理解数值稳定性和多阶段 Reduction。不要直接计算 `exp(x)` 后求和，否则输入较大时容易溢出。

### Day 6：统一 Benchmark 与 Profiling

统一 Benchmark 规则：

1. 每个 Shape 先 Warm-up 10～20 次；
2. 使用 CUDA Event 测量，至少重复 50～100 次；
3. 在读取 Event 结果前显式同步；
4. Naive、Optimized、PyTorch Reference 使用相同输入、dtype、Shape 和 GPU；
5. 记录 median、mean 或 min latency，避免只报告一次运行结果；
6. Benchmark 时不要设置 `CUDA_LAUNCH_BLOCKING=1`，它只用于调试异步错误；
7. 每次实验记录 GPU 是否有其他进程，以及显存占用情况。

建议的 Shape 覆盖方式：

- `rows`：从小 Batch/Sequence 到较大 Batch/Sequence，例如 `1、8、32、128、512`；
- `hidden_size`：至少覆盖 `768、1024、2048、4096`，有余力再测试 `8192`；
- dtype：FP32、FP16，A100 上补充 BF16；
- 同时覆盖小维度、典型模型维度和大维度，避免只在单一 Shape 上优化。

建议记录以下指标：

| 算子 | 必须记录 | 可选指标 |
|---|---|---|
| RMSNorm | latency、有效 GB/s、误差 | TFLOP/s、寄存器、Occupancy |
| Softmax | latency、有效 GB/s、误差 | 估算 FLOP/s、Warp Stall、Shared Memory |

带宽估算至少要说明统计了哪些读写：输入、权重和输出是否各算一次，是否存在额外中间缓冲。不同实现的统计口径必须一致。

Nsight 分析至少完成一次：

- 用 Nsight Systems 观察 CPU/GPU 时间线、Kernel Launch 和同步；
- 用 Nsight Compute 观察 Global Memory Throughput、Achieved Occupancy、寄存器使用、Shared Memory、Warp Stall 和指令吞吐；
- 选择一个主要瓶颈，例如非合并访存、过多同步、Reduction 依赖或 Launch Overhead；
- 用 Profile 结果验证优化假设，而不是只根据代码直觉下结论。

### Day 7：整理报告与面试复盘

完成一份 2～3 页的 Profiling 与瓶颈说明，至少包括：

1. 实验机器、软件版本和测试 Shape；
2. RMSNorm/Softmax 的计算公式和线程映射；
3. Reference、Naive、Optimized 三种版本的正确性结果；
4. 优化前后性能表和至少一张图；
5. Nsight 关键指标截图或摘要；
6. 一个主要瓶颈的证据链；
7. 一项有效优化、一项无效或收益有限的优化；
8. 当前实现的边界条件和下一步改进。

最后用 5～10 分钟口头回答：

- RMSNorm 为什么通常更接近 Memory-bound？
- 为什么需要 FP32 Accumulation？
- Warp Shuffle 相比 Shared Memory Reduction 减少了什么开销？
- 向量化 Load/Store 什么时候会失效或不安全？
- 为什么更高 Occupancy 不一定带来更高性能？
- 如何计算一个 Kernel 的算术强度？
- CUDA Event 为什么比普通 CPU 计时更适合测 Kernel？
- 一个优化如果只在单一 Shape 上有效，是否可以称为成功？

## 5. 推荐实验组织

建议将本周代码按以下逻辑组织。目录名称可以根据实际实现调整，但 Reference、CUDA 实现、Benchmark 和报告应分开：

```text
week1/
├── vector_add/
├── reduction/
├── rmsnorm/
│   ├── reference.py
│   ├── naive.cu
│   ├── optimized.cu
│   ├── test_correctness.py
│   └── benchmark.py
├── softmax/
│   ├── reference.py
│   ├── naive.cu
│   ├── optimized.cu
│   ├── test_correctness.py
│   └── benchmark.py
├── profiling/
└── report.md
```

每次实验至少保存：命令行、GPU 编号、Shape、dtype、Warm-up 次数、重复次数、原始结果和代码版本。不要只保存最终平均值。

## 6. 本周验收清单

### 必须完成

- [ ] 能解释 Thread/Warp/Block/Grid/SM/SIMT；
- [ ] 完成 Vector Add 和 Reduction 基础练习；
- [ ] 完成 PyTorch Reference、Naive CUDA、Optimized CUDA RMSNorm；
- [ ] 完成 PyTorch Reference、Naive CUDA、Optimized CUDA Softmax，或在进度不足时至少完成正确的 CUDA Softmax 和基础 Benchmark；
- [ ] RMSNorm 和 Softmax 支持至少 FP32、FP16 中的两种 dtype；
- [ ] 完成多组 `rows × hidden_size` 的正确性测试；
- [ ] 使用 CUDA Event，而不是未同步的 CPU wall-clock 作为主要延迟数据；
- [ ] 报告 latency、有效带宽和误差；
- [ ] 至少对一个 Kernel 完成 Nsight 分析；
- [ ] 写出 2～3 页 Profiling 与瓶颈说明。

### 面试级验收

- [ ] 能从零画出 RMSNorm 的线程/Block 映射；
- [ ] 能解释一次 Warp Reduction 的数据流；
- [ ] 能说明一个优化为什么有效，以及它的适用边界；
- [ ] 能从访存量、FLOPs 和时间估算算术强度；
- [ ] 能用 Nsight 指标支持瓶颈结论，而不是只引用最终速度；
- [ ] 能解释为什么某个更高 Occupancy 的版本可能更慢；
- [ ] 能在 10 分钟内完整讲解一个 Kernel 的设计、验证、Benchmark 和优化过程。

## 7. 本周结束时的交付物

1. 两个可运行、可测试的 CUDA Kernel：RMSNorm 和 Softmax；
2. 一张包含多组 Shape、dtype、Reference/Naive/Optimized 的性能表；
3. 一份 2～3 页的 Nsight Profiling 与瓶颈分析笔记；
4. 一份记录正确性容差、Benchmark 方法和已知限制的 README 或实验记录。
