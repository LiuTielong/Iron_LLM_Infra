import math
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def attention_prefill(q, k, v):
    # q, k, v: [batch, heads, seq_len, head_dim]
    batch, heads, seq_len, head_dim = q.shape

    scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=q.device,
            dtype=torch.bool,
        ),
        diagonal=1
    )

    scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)

    output = torch.matmul(probs, v)

    return output


def attention_decode(q_new, k_cache, v_cache, k_new, v_new):
    # q_new, k_new, v_new: [B, H, 1, D]
    # k_cache, v_cache:    [B, H, T_past, D]

    k_all = torch.cat([k_cache, k_new], dim=2)
    v_all = torch.cat([v_cache, v_new], dim=2)

    scale = 1.0 / math.sqrt(q_new.shape[-1])

    scores = torch.matmul(
        q_new,
        k_all.transpose(-2, -1),
    ) * scale

    probs = torch.softmax(scores, dim=-1)

    output = torch.matmul(probs, v_all)

    return output, k_all, v_all


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 GPU 节点、支持 CUDA 的 PyTorch 环境中运行。")

    torch.manual_seed(0)
    # FP32 正确性测试关闭 TF32，减少不同矩阵乘法路径的精度差异。
    torch.backends.cuda.matmul.allow_tf32 = False

    batch, heads, seq_len, head_dim = 1, 4, 128, 64
    decode_steps = 8
    prompt_len = seq_len - decode_steps
    shape = (batch, heads, seq_len, head_dim)

    q = torch.randn(shape, device="cuda", dtype=torch.float32)
    k = torch.randn(shape, device="cuda", dtype=torch.float32)
    v = torch.randn(shape, device="cuda", dtype=torch.float32)

    print(f"shape={shape}, dtype={q.dtype}")

    # 1. 完整序列的 causal Prefill，与 PyTorch Math Attention 比较。
    prefill_out = attention_prefill(q, k, v)

    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )

    torch.testing.assert_close(prefill_out, ref, atol=1e-5, rtol=1e-5)
    prefill_error = (prefill_out - ref).abs().max().item()
    print(f"Prefill PASS, max_abs_error={prefill_error:.6e}")

    # 2. 先处理前 120 个 Token，将它们的 K/V 作为初始缓存。
    k_cache = k[:, :, :prompt_len, :].clone()
    v_cache = v[:, :, :prompt_len, :].clone()
    prompt_out = attention_prefill(q[:, :, :prompt_len, :], k_cache, v_cache)

    torch.testing.assert_close(
        prompt_out, ref[:, :, :prompt_len, :], atol=1e-5, rtol=1e-5
    )

    # 3. 逐个处理剩余 Token，每次都使用上一步返回的 KV Cache。
    max_decode_error = 0.0

    for pos in range(prompt_len, seq_len):
        decode_out, k_cache, v_cache = attention_decode(
            q[:, :, pos:pos + 1, :],
            k_cache,
            v_cache,
            k[:, :, pos:pos + 1, :],
            v[:, :, pos:pos + 1, :],
        )

        # 因果 Attention 不看未来，逐步 Decode 应与完整 Prefill 对齐。
        torch.testing.assert_close(
            decode_out, prefill_out[:, :, pos:pos + 1, :],
            atol=1e-5, rtol=1e-5,
        )
        error = (decode_out - prefill_out[:, :, pos:pos + 1, :]).abs().max().item()
        max_decode_error = max(max_decode_error, error)

        # assert_close 同时检查形状和数值；缓存拼接应精确保留原数据。
        torch.testing.assert_close(
            k_cache, k[:, :, :pos + 1, :], atol=0, rtol=0
        )
        torch.testing.assert_close(
            v_cache, v[:, :, :pos + 1, :], atol=0, rtol=0
        )

    print(f"Decode ({decode_steps} steps) PASS, max_abs_error={max_decode_error:.6e}")
    print(f"KV Cache PASS, length: {prompt_len} -> {k_cache.shape[2]}")
    print("全部功能验证通过")


if __name__ == "__main__":
    main()
