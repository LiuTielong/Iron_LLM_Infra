# Week 2：Softmax 与 FlashAttention

## 主要目标

完成Softmax CUDA 算子的编写和实现。然后理解和复现 FlashAttention v1/v2 的核心算法。

本周以单卡 A100、前向计算为主。FP32 只用于先验证索引和算法；正式路径必须支持 FP16、BF16 输入，并使用 FP32 累加。A100 的 Tensor Core 应用于 Attention 中的 QKᵀ 和 PV 矩阵乘法。反向传播、GQA 以外的复杂变体放到后续优化。

## 库优先原则

本周不把“从头重写所有基础算子”作为目标。每个算子最多手写一个教学版本，然后用成熟库建立生产基线，再把自定义 Kernel 用在库无法直接覆盖的融合或调度场景。

本周需要熟悉的库和接口范围是：

| 组件 | 本周用途 |
|---|---|
| CUDA Runtime | Stream、Event、显存、Kernel Launch、错误检查 |
| CUB | Block/Warp Reduce，作为归约的库实现基线 |
| cuBLASLt | FP16/BF16/TF32 GEMM，使用 A100 Tensor Core |
| CUTLASS 或 WMMA | 理解 Tensor Core tile、布局和累加类型；二选一完成基本实验 |
| PyTorch Extension / ATen | Python Tensor、Stream 和自定义 CUDA Kernel 的连接 |
| cuDNN 或 PyTorch 算子 | Softmax 等成熟算子的参考实现，可选调用 |

Thrust 可以作为补充，但不把完整 Thrust API 列为本周验收内容。NCCL、cuSPARSE 和 cuFFT 与本周 Attention 主线关系较弱，后续再学。

## 学习步骤

### 第一步：独立实现稳定 Softmax

对二维输入 `[rows, cols]` 的最后一维实现：

```text
m = max(x)
s = sum(exp(x - m))
y = exp(x - m) / s
```

实现两个版本，并保持中间归约使用 FP32：

- Shared Memory Reduction 版本；
- Warp Shuffle Reduction 版本。

先支持 FP32；随后增加 FP16 和 BF16 输入。输入和输出可以是低精度，但最大值、指数和累加使用 FP32。手写版本完成后，再用 CUB 的 BlockReduce/WarpReduce 或 PyTorch Softmax 做库基线，比较接口、正确性和性能。不能只把 FP16/BF16 转成 FP32 后再测试。

自己生成随机数据，在 Python 中分别调用自己的 Kernel 和 `torch.softmax(x, dim=-1)`，比较最大绝对误差、NaN/Inf、每行输出和、CUDA Event 延迟和逻辑有效带宽。

至少测试普通输入和加上较大常数的输入，确认数值稳定性。这一步只验证 Softmax 本身，不需要接入模型。

### 第二步：建立标准 Attention 推理基线

先用 PyTorch 写一个最小 decoder-only Attention 流程。输入和中间张量使用：

```text
Q, K, V: [batch, heads, seq_len, head_dim]
```

实现：

```text
QKV Projection
→ QKᵀ / sqrt(head_dim)
→ causal mask
→ softmax
→ P @ V
→ output projection
```

先完成 Prefill：一次处理完整 Prompt。然后加入 Decode：每次只输入一个新 Token，并把历史 K/V 保存到 KV Cache。

这一版使用 PyTorch 算子，作为后续 FlashAttention 的正确性和性能基线。至少验证 Prefill 输出、Decode 每一步输出，以及 KV Cache 的追加和读取。

### 第三步：调用基础库并准备 Tensor Core 矩阵乘法

FlashAttention 的主要计算不是 Softmax，而是两个矩阵乘法：

```text
QKᵀ
P @ V
```

在接入 FlashAttention 前，先使用 cuBLASLt 完成一个小型 GEMM，使用 `head_dim=64` 和 FP16/BF16 输入、FP32 累加。记录矩阵描述、布局、workspace、CUDA Stream 和算法选择，确认 cuBLASLt 在 A100 上使用 Tensor Core。

然后再用 CUTLASS 或 WMMA 实现一个最小 tiled GEMM，用于理解 Tensor Core 的 tile、布局和 fragment；不要求这个手写版本超过 cuBLASLt。

这一阶段需要比较：

- 普通 FP32 CUDA FMA 版本；
- cuBLASLt Tensor Core 版本；
- CUTLASS/WMMA 教学版本；
- `torch.matmul` 结果。

确认编译目标为 `sm_80`，并用 Nsight Compute 或指令/吞吐指标确认确实执行了 Tensor Core，而不是只改变了输入 dtype。Tensor Core 版本的 tile、布局、对齐和累加类型要记录下来。

### 第四步：实现 FlashAttention v1 核心算法并接入 Prefill

先不要修改完整模型结构，只替换 Attention 中的 `QKᵀ → Softmax → P@V` 部分。

实现分块计算：

1. 一个 Query tile 与一个 K/V tile 计算分数 tile；
2. 不把完整的 `S` 或 `P` 矩阵写入 Global Memory；
3. 使用 Online Softmax 维护每一行的 `m、l、a`；
4. 所有 K/V tiles 处理完成后得到输出 `O = a / l`。

Online Softmax 更新公式：

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
p = exp(s - m_new)
l_new = alpha * l + sum(p)
a_new = alpha * a + p @ V_tile
```

先支持 `head_dim=64`、无 dropout。第一版可以调用 cuBLASLt 完成 QKᵀ 和 PV，自己实现 Online Softmax；下一版再尝试把矩阵乘法和 Softmax 融合到自定义 Kernel。这样可以分别观察成熟库 GEMM、融合收益和中间结果写回的代价。Online Softmax 的 `m、l、a` 使用 FP32。把这个实现接入第二步的 Prefill Attention，并与 PyTorch 基线比较输出和显存。

### 第五步：实现 FlashAttention v2 的工作划分并接入同一流程

在相同 Attention 接口下逐步改变并行组织：

1. Grid 覆盖 `(batch, head, query_tile)`，让不同 Query tile 由不同 Block 并行处理；
2. 使用未归一化的 `a` 累加，最后统一除以 `l`，减少中间归一化；
3. 在 Block 内让不同 Warp 负责不同 Query 行，减少跨 Warp 的部分和合并。

这里要明确 v1 与 v2 的差异来自工作划分和累加方式，而不是只改 Block Size 或调用一次 Warp Shuffle。

先比较 Prefill，再把同一个 Attention 接口用于 Decode。Decode 可以先使用简单的 KV Cache 布局，不需要立即实现 Paged KV Cache。

### 第六步：接入真实 LLM 推理流程

最小 decoder-only 流程通过后，选择服务器上一个较小、容易控制的模型，优先使用 Llama-2 7B；确认接口后再接入 Qwen-3。不要一开始直接修改整个生成系统，按以下顺序推进：

1. 加载 Hugging Face 模型和 tokenizer，固定一段 Prompt、采样参数和随机种子。
2. 先只替换一个 Transformer Layer 的 Attention，保留模型原有的 Embedding、RMSNorm、RoPE、QKV Projection、Output Projection 和权重。
3. 对同一批 Token 比较原始 Attention 与自写 Kernel 的 Layer 输出和最终 logits。
4. 扩展到所有 Layer 的 Prefill，比较 TTFT 和峰值显存。
5. 再处理 Decode：读取并追加模型的 KV Cache，比较每个新 Token 的输出和 TPOT/ITL。
6. 最后运行短文本生成，确认替换 Kernel 后生成结果在允许的数值误差内一致。

真实模型接入时，模型本身继续负责：

```text
Tokenizer、Embedding、RoPE、QKV Projection、RMSNorm、Sampling
```

你的 Kernel 只替换 Attention 的核心部分：

```text
Q、K、V + causal 语义 → Attention Output
```

Llama-2 可以作为第一目标。Qwen-3 通常涉及 GQA，Query Head 数和 KV Head 数不同；接入 Qwen-3 前需要让 Kernel 支持 GQA，或在 Python 侧临时展开 K/V 作为正确性基线，并记录额外显存成本。

真实模型的 Prefill 和 Decode 必须分开记录。只在 Prefill 中运行自写 Kernel，Decode 暂时使用 PyTorch，也可以作为中间结果，但不能称为完整的端到端替换。

### 第七步：统一验证和性能分析

对以下实现使用相同输入、模型参数和推理流程：

- PyTorch 显式 Attention 基线；
- 自写 FlashAttention v1；
- 自写 FlashAttention v2；
- 可选：PyTorch SDPA 的指定后端或官方 FlashAttention。

至少记录 Prefill 延迟、Decode 单 Token 延迟、端到端输出误差、峰值显存、Attention 的序列长度和 head 数。

建议先固定：`batch=1、heads=8、head_dim=64`，再测试 `seq_len=128、512、1024、2048`。所有性能测试使用 CUDA Event，先 Warm-up，再重复多次；编译、显存分配和正确性检查不计入 Kernel 时间。

重点解释：Softmax 为什么需要减最大值；Online Softmax 为什么必须重缩放旧状态；FlashAttention 为什么减少显存读写和中间矩阵占用；v2 如何增加 Query 并行度、减少同步和部分和合并；为什么教学版 FP32 Kernel 可能显存更省，却比 PyTorch/官方库更慢。

## 建议目录

```text
week2/
├── Softmax.md
├── softmax/
│   ├── softmax_ext.cu
│   └── test_softmax.py
├── attention/
│   ├── model.py              # 最小 decoder-only 推理流程与 KV Cache
│   ├── attention_reference.py
│   ├── flash_v1_ext.cu
│   ├── flash_v2_ext.cu
│   └── test_attention.py
└── results/
    └── report.md
```

## 验收标准

- [ ] Softmax 两个 CUDA 版本与 `torch.softmax` 对齐。
- [ ] Softmax 支持 FP32、FP16、BF16，并记录 FP32 累加误差。
- [ ] 有稳定 Softmax 的误差、延迟和带宽记录。
- [ ] 能使用 CUB 完成一次 Block/Warp Reduction，并与手写 Reduction 比较。
- [ ] 能从 PyTorch Extension 中调用 cuBLASLt 完成 FP16/BF16 GEMM。
- [ ] 有一个使用 CUTLASS 或 WMMA 的 Tensor Core 矩阵乘法实验。
- [ ] 确认 Tensor Core 版本在 A100 上实际执行 Tensor Core 指令。
- [ ] 有一个能执行 Prefill 和 Decode 的最小 decoder-only 推理流程。
- [ ] FlashAttention v1 接入 Prefill，并使用 Online Softmax，不保存完整注意力矩阵。
- [ ] FlashAttention v2 改变 Query/Block/Warp 工作划分，并接入同一流程。
- [ ] FlashAttention 的 QKᵀ 和 PV 使用 FP16/BF16 Tensor Core，统计量使用 FP32。
- [ ] v1/v2 都通过 Prefill 正确性测试，至少有一个版本接入 Decode + KV Cache。
- [ ] 在至少一个真实 Llama-2 或 Qwen-3 模型中替换一个 Attention Layer，并通过 hidden state/logits 对比。
- [ ] 将替换扩展到真实模型的 Prefill；有余力时完成 Decode + KV Cache 和短文本生成。
- [ ] 有延迟、峰值显存和误差对比，并写出性能差异的原因。

本周先从第一步开始：独立完成稳定 Softmax，再与 `torch.softmax` 做对比。
