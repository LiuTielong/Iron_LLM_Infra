import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# 当前实验硬件为 A100
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

source_dir = Path(__file__).resolve().parent

official_dir = (
    source_dir.parents[1] / "flash-attention-v1" / "csrc" / "flash_attn"
)

ext = load(
    name="flash_v1_myself_ext",
    sources=[str(source_dir / "flash_v1_ext.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3", "-lineinfo",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=True,
    extra_include_paths=[
        str(official_dir / "src"),
        str(official_dir / "cutlass" / "include"),
    ],
)

@torch.inference_mode()
def main():
    torch.manual_seed(0)

    shape = (1, 512, 32, 128)

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(shape, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = ext.forward(q, k, v)
        torch.cuda.synchronize()

        assert out.shape == q.shape
        assert out.dtype == q.dtype
        assert out.device == q.device

        print(f"接口与启动通过：shape={tuple(out.shape)}, dtype={out.dtype}")

    print("骨架验证通过；kernel 尚未执行 Attention 计算。")

if __name__ == "__main__":
    main()