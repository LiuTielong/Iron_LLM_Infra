from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# 1. 编译并加载CUDA 扩展
ext = load(
    name="softmax_ext", 
    sources=[str(Path(__file__).with_name("softmax_ext.cu"))],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)

# 2. 准备GPU输入
torch.manual_seed(0)

x = torch.randn(128, 1024, device="cuda", dtype=torch.float32)

# 3. 功能测试
for name, data in [("random", x), ("large values", x + 1000)]:
    y = ext.softmax(data)
    ref = torch.softmax(data, dim=-1)

    torch.testing.assert_close(
        y, ref, atol=1e-5, rtol=1e-5
    )

    # 每行概率和应该接近1
    row_sum = y.sum(dim=-1)
    torch.testing.assert_close(
        row_sum, torch.ones_like(row_sum), atol=1e-5, rtol=1e-5
    )

    max_error = (y - ref).abs().max().item()
    print(f"{name} max error: {max_error:.3g}")

# 4. CUDA Event 计时函数
@torch.inference_mode()
def benchmark(fn, warmpup=20, repeats=100):
    for _ in range(warmpup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeats):
        fn()

    stop.record()

    stop.synchronize()

    return start.elapsed_time(stop) / repeats


# 5. 使用相同输入比较性能
custom_ms = benchmark(lambda: ext.softmax(x))
torch_ms = benchmark(lambda: torch.softmax(x, dim=-1))

# 逻辑访存量：输入读取一次，输出写入一次
logical_bytes = 2 * x.numel() * x.element_size()

print(f"\nshape={tuple(x.shape)}, dtype={x.dtype}")
print(f"{'implementation':<18} {'avg_ms':>10} {'GB/s':>10}")

for name, ms in [("custom", custom_ms), ("torch.softmax", torch_ms)]:
    bandwidth = logical_bytes / (ms * 1e6)
    print(f"{name:<18} {ms:>10.6f} {bandwidth:>10.2f}")

print(f"speedup = {torch_ms / custom_ms:.2f}x")