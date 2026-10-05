from torch.utils.cpp_extension import load
import torch

ext = load(
    name="gemm_ext",
    sources=["gemm_ext.cu"],
    verbose=True,
)

# 准备GPU输入
M, K, N = 1024, 1024, 1024

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

# 功能测试
out = ext.gemm_forward(a, b)
ref = torch.matmul(a.float(), b.float())

max_error = torch.max(torch.abs(out - ref))

print("output shape:", out.shape)
print("max_abs_error:", max_error)
print("correct:", torch.allclose(out, ref, atol=1e-2, rtol=1e-2))

# CUDA Event 计时函数
@torch.inference_mode()
def benchmark(fn, warmup=20, repeats=500):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeats):
        fn()

    end.record()
    end.synchronize()

    return start.elapsed_time(end) / repeats

# 比较性能
custom_ms = benchmark(lambda: ext.gemm_forward(a, b))
torch_ms = benchmark(lambda: torch.matmul(a, b))

logical_bytes = a.numel() * a.element_size() \
                + b.numel() * b.element_size() + out.numel() * out.element_size()


for name, ms in [("custom", custom_ms), ("torch.matmul", torch_ms)]:
    bandwidth = logical_bytes / (ms * 1e6)
    print(f"{name:<18} {ms:>10.6f} {bandwidth:>10.2f}")

print(f"speedup: {torch_ms / custom_ms:.2f}x")