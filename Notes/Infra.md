# LLM Infra 面试学习计划

## 1. 学习目标

目标岗位要求同时覆盖 CUDA Kernel 与 LLM Serving。现有的 8×A100 40GB 足以完成所需实验，不必将重点放在训练超大模型，而应集中建立以下三项能力：

1. 能独立实现、调试和优化 CUDA 算子，并用性能数据解释优化效果；
2. 能分析 LLM 推理过程中的计算、显存、KV Cache、调度和通信瓶颈；
3. 能读懂 vLLM、SGLang 的核心执行路径，通过实验比较不同推理策略，并完成小型源码修改。

整个计划分为两个相互独立的阶段，每周按约 15～20 小时设计：

- **第一阶段：3 周核心阶段。** 结束后应形成足够用于面试的完整作品和知识闭环，即使不再进行第二阶段，也能独立讲解 CUDA 优化、LLM 推理原理以及 vLLM/SGLang 的关键设计。
- **第二阶段：3 周进阶阶段。** 仅在时间允许时继续，主要补充 Attention Kernel、多卡通信、高级推理优化和更深入的框架修改。

第一阶段的原则是“少而完整”：不追求覆盖所有技术点，而要把有限的实验做出正确性验证、性能数据、Profiling 分析和设计说明。

---

## 2. 需要掌握的知识

### 两阶段学习优先级

| 方向 | 第一阶段必须掌握 | 第二阶段进阶内容 |
|---|---|---|
| CUDA | 执行模型、内存层次、合并访存、Warp、Reduction、RMSNorm/Softmax、正确计时、Nsight | `cp.async`、Tensor Core、复杂 Fusion、Attention/Paged Attention Kernel |
| LLM 推理 | Prefill/Decode、KV Cache、GQA、Continuous Batching、Paged KV、Prefix Cache、Chunked Prefill | Speculative Decoding、量化、KV Offload、Prefill/Decode Disaggregation |
| vLLM/SGLang | 请求路径、Scheduler、Worker、KV Cache 管理、基础 Benchmark | 高级调度、分布式执行、源码级性能优化 |
| 多卡 | TP/DP 基本原理、AllReduce、硬件拓扑 | NCCL 系统实验、通信计算重叠、TP/DP Scaling |

### 2.1 CUDA 编程与性能优化

#### CUDA 执行模型

- Thread、Warp、Block、Grid、SM 与 SIMT；
- Warp Divergence 与分支效率；
- Block 大小、Occupancy 与寄存器压力；
- Kernel Launch 与异步执行语义。

#### GPU 存储层次

- Register、Shared Memory、L1/L2 Cache、Global Memory；
- 合并访存、向量化访存与数据对齐；
- Shared Memory Bank Conflict；
- 数据复用与 Shared Memory Tiling；
- `cp.async`、双缓冲以及计算和访存重叠。

#### 并行算法与数值计算

- Warp Shuffle、Reduction、Scan；
- Atomic、`__syncthreads()` 与同步语义；
- FP32、TF32、FP16、BF16、INT8/INT4；
- FP32 Accumulation 与数值稳定性；
- Tensor Core、WMMA 和 CUTLASS 的基本思想；
- CUDA Stream、Event 和 CUDA Graph。

#### 性能分析

需要熟练使用：

- Nsight Systems：分析 CPU/GPU 时间线、Kernel Launch、通信与计算重叠；
- Nsight Compute：分析访存、Occupancy、Warp Stall 和指令吞吐；
- PyTorch Profiler：分析算子级时间和显存；
- Roofline Model：判断 Kernel 是 Memory-bound 还是 Compute-bound。

对于每个 Kernel，都应能回答：

1. 理论 FLOPs 和访存量是多少？
2. 算术强度是多少？
3. 瓶颈位于 DRAM、Cache、Shared Memory、计算单元还是 Launch Overhead？
4. 为什么某项优化有效或无效？
5. 为什么更高的 Occupancy 不一定带来更高性能？

> A100 属于 Compute Capability 8.0，适合学习 BF16、Tensor Core 和异步 Global-to-Shared Memory Copy，但不具备 Hopper/Blackwell 的原生 FP8 Tensor Core，因此不必将 FP8 Kernel 作为学习重点。

### 2.2 Transformer 推理原理

#### 模型算子

- Embedding、Linear、RMSNorm；
- RoPE；
- QKV Projection；
- MHA、MQA、GQA；
- Causal Attention 与 FlashAttention；
- SwiGLU；
- Sampling：Temperature、Top-k、Top-p。

#### Prefill 与 Decode

- Prefill 一次处理输入 Prompt，通常具有较高计算密度，主要影响 TTFT；
- Decode 每轮为每个请求生成一个或少量 Token，通常更受模型权重和 KV Cache 访存限制；
- 长 Prompt、高并发、长输出对计算和显存的影响不同，不能只用单一 Tokens/s 衡量系统。

#### KV Cache

必须能够估算模型和 KV Cache 的显存占用：

```text
模型权重显存 ≈ 参数量 × 每个参数的字节数

每 Token 的 KV Cache：
2 × 层数 × KV Head 数 × Head Dimension × dtype 字节数
```

其中第一个 `2` 表示 Key 和 Value。如果 KV Head 在 Tensor Parallel Rank 之间分片，单卡 KV Cache 通常还需除以相应分片数。

需要理解：

- 连续内存 KV Cache 的浪费和碎片问题；
- Paged KV Cache、Block Table、Free List；
- Prefix Cache 与 Cache Eviction；
- MQA/GQA 对 KV Cache 容量和带宽的影响；
- KV Cache Quantization 与 Offloading。

#### 推理优化技术

- Continuous Batching；
- Chunked Prefill；
- Prefix Caching；
- Speculative Decoding；
- Weight-only Quantization 与 W8A8；
- Prefill/Decode Disaggregation；
- Kernel Fusion 与 CUDA Graph。

### 2.3 分布式推理

需要掌握以下并行方式：

| 并行方式 | 切分对象 | 主要通信 | 典型用途 |
|---|---|---|---|
| Data Parallelism | 请求/Batch | 通常较少 | 增加整体服务吞吐 |
| Tensor Parallelism | 单层矩阵 | AllReduce/AllGather | 单卡放不下模型或降低单请求延迟 |
| Pipeline Parallelism | 模型层 | Point-to-Point | 跨节点部署超大模型 |
| Expert Parallelism | MoE 专家 | All-to-All | MoE 模型推理 |

需要理解：

- NCCL 的 AllReduce、AllGather、ReduceScatter 和 All-to-All；
- Ring、Tree 等集合通信的基本思想；
- NVLink、NVSwitch、PCIe 和跨节点网络的区别；
- 通信延迟、带宽以及消息大小的关系；
- 通信与计算重叠；
- 为什么 Tensor Parallel Scaling 通常不是线性的。

实验开始前记录硬件拓扑：

```bash
nvidia-smi topo -m
```

### 2.4 vLLM 核心结构

至少能够追踪一次请求的完整执行路径：

```text
API Server
  → Engine Core
  → Scheduler
  → KVCacheManager
  → Worker / Model Runner
  → Attention Backend
```

重点问题：

- Waiting、Running、Finished 等请求状态如何变化？
- Scheduler 如何使用 Token Budget 组织一次执行？
- Continuous Batching 如何让请求动态加入和退出？
- Chunked Prefill 如何与 Decode 混合执行？
- KV Block 如何申请、映射、回收和复用？
- Prefix Cache 如何识别已经计算过的前缀？
- KV Cache 不足时如何抢占、重算或换出？
- Tensor Parallel Worker 如何协同执行模型？
- Attention Backend 和自定义 Kernel 如何接入？

### 2.5 SGLang 核心结构

建议按照以下路径阅读：

```text
Tokenizer Manager
  → Scheduler
  → Prefill / Decode Batch
  → TP Worker
  → Memory Pool
  → Radix Cache
```

重点问题：

- RadixAttention 如何用 Radix Tree 表示和复用公共前缀？
- LPM/Cache-aware Scheduling 如何提高缓存命中率？
- `ReqToTokenPool` 与 `TokenToKVPool` 分别解决什么问题？
- Prefill 和 Decode Batch 如何组织？
- OOM Retract 如何工作？
- CPU 调度与 GPU 执行如何重叠？
- 结构化输出和 Speculative Decoding 如何进入调度流程？

vLLM 和 SGLang 的能力已经有大量重叠。不要简单记忆“vLLM 等于 PagedAttention，SGLang 等于 RadixAttention”，而应比较二者在缓存索引、请求调度、运行时组织和具体 Workload 下的表现。

---

## 3. 动手项目

### 3.1 项目一：CUDA Kernel Lab

#### 第一阶段：完成两个可深入讲解的 Kernel

1. 用 Vector Add 和 Reduction 熟悉 CUDA，不将其作为主要作品；
2. 完整实现并优化 **RMSNorm**；
3. 完整实现并优化 **Softmax**；
4. 如果时间有余，增加 Fused Residual + RMSNorm。

每个算子至少包含：

```text
PyTorch Reference
→ Naive CUDA
→ Optimized CUDA
→ 可选 Triton 实现
```

每个算子的验收标准：

- 支持具有代表性的 FP32、FP16 或 BF16 输入；
- 与 PyTorch Reference 对比正确性和误差；
- 覆盖多组 Batch、Sequence Length 和 Hidden Size；
- 正确 Warm-up，并使用 CUDA Event 计时；
- 报告 Latency、GB/s 或 TFLOP/s；
- 使用 Nsight 解释一个主要瓶颈；
- 至少实现并验证一种实际有效的优化。

第一阶段优先练习：

- 向量化 Load/Store；
- Warp Shuffle Reduction；
- FP32 Accumulation；
- 减少同步和中间结果写回；
- 简单 Kernel Fusion。

#### 第二阶段：扩展到 Transformer 核心 Kernel

- RoPE 和 SwiGLU；
- Tiled GEMM；
- 简化版 Causal Attention；
- 简化版 Paged Decode Attention；
- `cp.async`、双缓冲和更复杂的 Kernel Fusion；
- 可选 Triton/CUTLASS 版本对照。

不要求手写 GEMM 超越 cuBLAS。能够解释性能差距和硬件瓶颈，比在单一 Shape 上偶然取得更高速度更有价值。

### 3.2 项目二：Mini LLM Serving Engine

#### 第一阶段：实现调度与 KV Cache 模拟器

第一阶段不要求完成一个高性能端到端模型引擎。用 Python 加少量 PyTorch 实现：

```text
请求队列与状态机
→ Continuous Batching
→ KV Block / Block Table / Free List
→ 请求结束后的 KV 回收
→ 简单 Prefix Cache
```

使用合成请求验证：

- 不同长度请求能够动态加入和退出 Batch；
- Block 不会重复分配，结束后能够正确回收；
- KV 不足时能够拒绝、等待或抢占，并说明所选策略；
- 公共前缀能够命中已有 Block；
- 输出一次完整的请求状态和 Block 映射变化。

#### 第二阶段：接入真实小模型

选择 0.5B～1.5B 模型，继续加入：

- 实际模型 Forward；
- Token Budget 和 Chunked Prefill；
- 流式输出；
- CUDA Graph 或自定义 Kernel；
- 与 Hugging Face 静态批处理进行性能比较。

项目重点不是击败 vLLM，而是能画出请求状态转换、KV Block 映射和调度过程，并解释设计取舍。

### 3.3 项目三：vLLM 与 SGLang 对照实验

#### 第一阶段：最小但完整的对照实验

统一使用一个 7B BF16 模型，先在单张 A100 上完成以下三组实验：

| 实验 | 控制变量 | 主要指标 |
|---|---|---|
| 并发扩展 | Concurrency=1/8/32/64 | TTFT、TPOT、吞吐、P99 |
| 公共前缀 | 无公共前缀与 2K 公共前缀 | Cache Hit、TTFT、吞吐 |
| 长短请求混合 | Chunked Prefill 开启/关闭 | 短请求 P99、Decode 抖动 |

第一阶段的重点不是覆盖大量参数，而是做到：

- 固定软件版本与完整配置；
- 使用完全相同的模型、Prompt 和采样参数；
- 运行 Warm-up 和多轮重复实验；
- 同时报告平均值、P50/P95/P99；
- 对结果给出机制解释，而不只给柱状图。

#### 第二阶段：扩大模型与优化范围

- 32B BF16：TP=2/4；
- 70B BF16：TP=8；
- TP 与 DP 的吞吐/延迟比较；
- AWQ/GPTQ 与 BF16 比较；
- 7B Target + 0.5B Draft 的 Speculative Decoding；
- 8K 公共前缀、长上下文及更复杂流量分布。

70B BF16 权重约占 140GB。TP=4 时单卡仅权重就接近 35GB，留给 KV Cache 和运行时的空间较少；在 40GB A100 上更适合采用 TP=8。

#### 公平比较原则

两个框架必须使用相同的：

- 模型和权重；
- dtype 和量化方式；
- 输入/输出 Token 分布；
- Sampling 参数；
- 最大并发数和 Request Rate；
- Prefix Cache、Chunked Prefill 等功能配置；
- GPU 数量、TP/DP 配置和测试时长。

需要报告：

- TTFT；
- TPOT/ITL；
- End-to-End Latency；
- Input/Output Tokens/s；
- P50/P95/P99；
- GPU 利用率和显存；
- Cache Hit Rate；
- 框架、模型、CUDA 和 Driver 版本。

不要得出“某框架永远更快”的结论。结论必须绑定到具体 Workload，例如：

> 在高前缀复用、长 Prompt、高并发、短输出的流量下，某种缓存与调度配置获得了更好的 TTFT 和总体吞吐。

### 3.4 项目四：NCCL 与多卡通信

#### 第一阶段：只做基础验证

- 使用 `nvidia-smi topo -m` 记录 8 卡拓扑；
- 使用 `nccl-tests` 测一次 2/4/8 GPU AllReduce；
- 观察小消息和大消息的 Latency/Bandwidth 差异；
- 能解释 AllReduce 为什么出现在 Tensor Parallel Linear 中。

#### 第二阶段：系统通信实验

使用 `nccl-tests` 系统测量：

- AllReduce；
- AllGather；
- ReduceScatter；
- All-to-All；
- 2/4/8 GPU；
- 多种消息大小。

随后实现两个小实验：

1. GEMM 后接 AllReduce，模拟 Row-Parallel Linear；
2. 使用不同 CUDA Stream，将通信与下一阶段计算重叠。

实验后应能解释：

- 小消息为什么更受 Latency 限制；
- 大消息为什么更受 Bandwidth 限制；
- 小模型使用 TP=8 为什么可能比单卡更慢；
- NVLink/NVSwitch/PCIe 拓扑如何影响扩展效率；
- 哪些情况下应增加 DP，而不是继续增加 TP。

### 3.5 项目五：框架源码修改

#### 第一阶段：完成一个低风险、可验证的小修改

从下列任务中选择一个：

- 为 vLLM 增加 KV Block 使用率或 Cache Hit 统计；
- 统计 SGLang Radix Cache 的命中长度分布；
- 增加一项 Scheduler 调试日志或 Prometheus 指标；
- 为已有调度边界条件补充单元测试。

#### 第二阶段：完成更深入的性能修改

- 为 vLLM 增加简单调度策略；
- 修改 Chunked Prefill Token Budget，分析延迟变化；
- 为 SGLang 增加简单请求排序策略；
- 将自己的 RMSNorm 或 RoPE Kernel 接入一个框架；
- 修复一个小 Bug，并补充测试。

源码修改的验收标准：

1. 能定位请求执行路径和修改位置；
2. 有单元测试或正确性验证；
3. 有修改前后的性能或行为数据；
4. 能说明修改影响的状态、边界条件和潜在副作用。

---

## 4. 两阶段执行安排

### 4.1 第一阶段：3 周核心准备

第一阶段结束后即按“可以参加面试”验收，不依赖第二阶段补齐成果。

#### 第 1 周：CUDA Kernel 与性能分析

学习内容：

- CUDA 执行模型、Warp、内存层次和同步；
- 合并访存、Warp Reduction、FP32 Accumulation；
- CUDA Event 正确计时；
- Nsight Systems/Compute 基础使用。

实践顺序：

1. 用 Vector Add 和 Reduction 完成环境及基础训练；
2. 实现 PyTorch Reference、Naive CUDA 和 Optimized CUDA 版 RMSNorm；
3. 实现并优化 Softmax；
4. 对多组 Shape 做正确性和性能测试；
5. 选择其中一个 Kernel 完成 Nsight 分析。

本周必须交付：

- 两个可运行、可测试的 CUDA Kernel；
- 一张优化前后性能表；
- 一份 2～3 页的 Profiling 与瓶颈说明。

如果进度落后，保留 RMSNorm 的完整优化和分析，Softmax 可以只做到正确、可 Benchmark，但不能省略正确性测试。

#### 第 2 周：LLM 推理原理与 vLLM/SGLang

学习内容：

- Transformer 推理算子；
- Prefill、Decode、KV Cache 和显存估算；
- Continuous Batching、Paged KV、Prefix Cache、Chunked Prefill；
- vLLM/SGLang 的请求路径、Scheduler 和 Worker。

实践顺序：

1. 启动同一个 7B BF16 模型的 vLLM 和 SGLang 服务；
2. 从 API 入口追踪到 Scheduler、KV Cache 和 Worker；
3. 完成并发扩展实验；
4. 完成 2K 公共前缀实验；
5. 完成长短请求混合及 Chunked Prefill 实验；
6. 整理公平比较方法和结果解释。

本周必须交付：

- 一张 vLLM/SGLang 请求路径对照图；
- 一套可复现 Benchmark 脚本与固定配置；
- TTFT、TPOT、吞吐、P50/P95/P99 图表；
- 一份不超过 5 页的实验结论。

#### 第 3 周：Mini Infra、源码修改与面试整理

学习内容：

- 请求状态机和 Token Budget；
- KV Block、Block Table、Free List 和 Prefix Cache；
- TP/DP 与 AllReduce 的基本原理；
- 阅读和修改大型推理框架源码的方法。

实践顺序：

1. 实现 Mini Continuous Batching Scheduler；
2. 实现 Paged KV Block Allocator 和 Prefix Cache；
3. 用合成请求展示状态转换和 Block 复用；
4. 在 vLLM 或 SGLang 中增加一项统计、日志或测试；
5. 运行一次 2/4/8 GPU AllReduce 基础测试；
6. 整理 README、架构图、结果和 10 分钟讲稿；
7. 针对高频问题做一次模拟面试。

本周必须交付：

- 一个可运行的 Mini Scheduler/KV Cache 项目；
- 一个有正确性验证的框架 Patch；
- 一份基础多卡拓扑和 AllReduce 数据；
- 一套可以在面试中演示和讲解的作品集。

### 4.2 第一阶段的时间分配原则

第一阶段总时间建议按以下比例分配：

| 内容 | 时间占比 |
|---|---:|
| CUDA 实现、调试与 Profiling | 35% |
| vLLM/SGLang 阅读与 Benchmark | 35% |
| Mini Scheduler/KV Cache | 20% |
| 源码 Patch、报告和模拟面试 | 10% |

如果总时间不足，应按以下顺序保留成果：

1. RMSNorm CUDA 的完整实现、优化和 Profiling；
2. vLLM/SGLang 的三组公平对照实验；
3. Mini Paged KV Cache 与 Continuous Batching；
4. 框架统计/日志 Patch；
5. Softmax 深度优化和 NCCL 扩展实验。

### 4.3 第二阶段：3 周可选进阶

#### 第 4 周：Attention 与高级 CUDA

- 实现 RoPE、SwiGLU；
- 实现简化 Causal Attention；
- 实现简化 Paged Decode Attention；
- 学习 `cp.async`、双缓冲、Tensor Core 和 CUTLASS/Triton；
- 比较 Prefill 与 Decode Shape 下的 Kernel 行为。

交付物：Attention/Paged Attention Kernel、Nsight 报告和性能对照。

#### 第 5 周：多卡推理与高级优化

- 系统测量 NCCL Collective；
- 做 TP=1/2/4/8 Scaling；
- 比较 32B/70B 模型的 TP 与 DP；
- 测试通信计算重叠；
- 测试量化和 Speculative Decoding。

交付物：TP/DP Scaling 曲线、通信分析和高级推理优化报告。

#### 第 6 周：框架深入与端到端项目

- 将 Mini Serving Engine 接入 0.5B～1.5B 真实模型；
- 加入 Chunked Prefill、流式输出和性能指标；
- 修改调度策略或接入自定义 CUDA Kernel；
- 完成更深入的框架 Patch 或尝试提交 PR；
- 补充系统设计和故障分析问题。

交付物：端到端 Mini Engine、深入 Patch/PR 和最终技术报告。

---

## 5. 面试验收清单

### 5.1 第一阶段结束时必须通过

#### CUDA

- [ ] 能从零写出 Reduction、Softmax、RMSNorm 中至少两个；
- [ ] 能正确使用 CUDA Event 测量异步 Kernel；
- [ ] 能分析合并访存、Bank Conflict、Warp Divergence 和寄存器压力；
- [ ] 能解释 Memory-bound 与 Compute-bound；
- [ ] 能用 Nsight 找到一个真实瓶颈；
- [ ] 能解释 Kernel Fusion 的收益和限制。

#### LLM 推理

- [ ] 能推导模型权重与 KV Cache 显存占用；
- [ ] 能解释 Prefill 与 Decode 的不同瓶颈；
- [ ] 能解释 Paged KV Cache 的收益和额外开销；
- [ ] 能解释 FlashAttention 为什么减少 HBM 访问；
- [ ] 能说明 MHA、MQA、GQA 对 KV Cache 的影响；
- [ ] 能解释 Continuous Batching、Prefix Cache 和 Chunked Prefill。

#### vLLM 与 SGLang

- [ ] 能画出两个框架的一次请求执行路径；
- [ ] 能解释 Scheduler、Worker 和 KV Cache Manager 的职责；
- [ ] 能比较 vLLM Prefix Cache 与 SGLang Radix Cache；
- [ ] 能解释不同调度策略如何影响吞吐和 Tail Latency；
- [ ] 能完成一个源码修改并给出数据；
- [ ] 能设计公平、可复现的对照实验。

#### Mini Infra 与多卡基础

- [ ] 能现场画出请求状态机和 KV Block Table；
- [ ] 能解释 Block 的分配、复用、回收和不足时的处理；
- [ ] 能解释 TP、DP 以及 AllReduce 的基本作用；
- [ ] 能根据硬件拓扑解释基础 NCCL 测试结果。

### 5.2 第二阶段进阶验收

- [ ] 能解释 TP、DP、PP 和 EP；
- [ ] 能说明常见 NCCL Collective 的语义；
- [ ] 能解释 TP Scaling 非线性的原因；
- [ ] 能根据模型大小、并发和拓扑选择 TP/DP；
- [ ] 能分析通信与计算是否有效重叠。
- [ ] 能实现或深入解释 FlashAttention/Paged Attention Kernel；
- [ ] 能解释 Speculative Decoding、量化和 Prefill/Decode Disaggregation；
- [ ] 能完成调度策略或自定义 Kernel 级别的框架修改。

---

## 6. 最终作品集

建议将成果整理成以下目录：

```text
llm-infra-interview/
├── cuda-kernels/
├── mini-serving-engine/
├── benchmarks-vllm-sglang/
├── nccl-experiments/
├── framework-patches/
└── report/
```

### 第一阶段完成后必须具备

最终至少准备四个可以在面试中深入讲解的案例：

1. 两个经过 Profiling 和优化的 CUDA Kernel；
2. 一个实现 Continuous Batching 和 Paged KV Cache 的 Mini Scheduler；
3. 一组可复现的 vLLM/SGLang 对照实验；
4. 一次 vLLM 或 SGLang 源码修改。

这四项已经构成独立、完整的第一阶段作品集，不需要依赖第二阶段成果才能用于面试。

### 第二阶段可增加

- Attention/Paged Attention CUDA Kernel；
- 系统化 NCCL 与 TP/DP Scaling 实验；
- 接入真实模型的 Mini Serving Engine；
- 调度策略、自定义 Kernel 或可提交上游的 Patch/PR。

每个案例都按照以下结构准备：

```text
问题背景
→ 系统或算子设计
→ 正确性验证
→ Benchmark 方法
→ 性能数据
→ 瓶颈分析
→ 优化与取舍
→ 后续改进
```

---

## 7. 推荐资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [vLLM Documentation](https://docs.vllm.ai/en/stable/)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Benchmark Guide](https://docs.sglang.ai/developer_guide/bench_serving)
- FlashAttention、PagedAttention、SGLang/RadixAttention 原始论文
