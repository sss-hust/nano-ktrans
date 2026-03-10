import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nano_ktrans.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr, D: tl.constexpr
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )

class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        三种注意力模式：
        1. 标准 Prefill:   flash_attn_varlen_func（全序列因果注意力）
        2. Chunked Prefill: flash_attn_with_kvcache + k/v 追加（分块 + cache 累积）
        3. Decode:          flash_attn_with_kvcache（单 token 查询缓存）

        输出统一为 [N, num_heads, head_dim] 3D 张量。
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        has_cache = k_cache.numel() > 0 and v_cache.numel() > 0

        if context.is_prefill:
            if context.is_chunked_prefill and has_cache:
                # ---- Chunked Prefill ----
                # flash_attn_with_kvcache 的 k=/v= 参数会自动将新 KV 追加到 cache，
                # 并在注意力计算中包含所有已缓存 + 新追加的 KV，实现跨 chunk 因果注意力。
                o = flash_attn_with_kvcache(
                    q.unsqueeze(0),           # [1, chunk_len, nheads, hdim]
                    k_cache, v_cache,
                    k=k.unsqueeze(0),         # 新 chunk 的 K，追加到 cache
                    v=v.unsqueeze(0),         # 新 chunk 的 V，追加到 cache
                    cache_seqlens=context.cache_seqlens,
                    softmax_scale=self.scale,
                    causal=True,
                ).squeeze(0)                  # → [chunk_len, nheads, hdim]
            else:
                # ---- 标准全量 Prefill ----
                if has_cache:
                    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables
                )
        else:
            # ---- Decode（单 token 逐步生成） ----
            if has_cache:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),               # [batch, 1, nheads, hdim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(1)                      # → [batch, nheads, hdim]

        return o
