from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# 编译并加载扩展；后续运行会复用编译缓存
ext = load(
    name="rmsnorm_ext",
    sources=[str(Path(__file__).with_name("rmsnorm_ext.cu"))],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)

torch.manual_seed(0)

# 直接在 GPU 上准备数据
x = torch.randn(128, 1024, device="cuda", dtype=torch.float32)
weight = torch.randn(1024, device="cuda", dtype=torch.float32)
eps = 1e-5

# 调用自己的 CUDA Kernel
y = ext.forward(x, weight, eps)

# PyTorch Reference：FP32 平方和与归一化
mean_sq = x.square().mean(dim=-1, keepdim=True)
ref = x * torch.rsqrt(mean_sq + eps) * weight

# 正确性验证
torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)

print("max_abs_error =", (y - ref).abs().max().item())
print("PyTorch 调用和正确性测试通过")