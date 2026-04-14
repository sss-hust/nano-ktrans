import torch
import torch.nn.functional as F
from torch import nn

from nano_ktrans.utils.context import get_context

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

HAS_TRITON = triton is not None and tl is not None
HAS_FLASH_ATTN = flash_attn_varlen_func is not None and flash_attn_with_kvcache is not None


if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr, key_stride,
        value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr,
        slot_mapping_ptr, D: tl.constexpr
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return

        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)

        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
):
    if slot_mapping is None:
        return

    if HAS_TRITON and key.is_cuda:
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
        return

    valid = slot_mapping >= 0
    if not torch.any(valid):
        return

    slots = slot_mapping[valid].to(torch.long)
    k_cache[0, slots] = key[valid]
    v_cache[0, slots] = value[valid]

class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == self.num_heads:
            return tensor
        repeat_factor = self.num_heads // tensor.shape[1]
        return tensor.repeat_interleave(repeat_factor, dim=1)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q_t.dtype).unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def _offset_causal_mask(
        self,
        query_len: int,
        key_len: int,
        *,
        prefix_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_positions = torch.arange(prefix_len, prefix_len + query_len, device=device).unsqueeze(1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        mask = torch.full((query_len, key_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask[key_positions <= query_positions] = 0.0
        return mask

    def _forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        has_cache = k_cache.numel() > 0 and v_cache.numel() > 0

        if context.is_prefill:
            if has_cache and context.slot_mapping is not None:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

            if context.is_chunked_prefill and has_cache:
                chunk_start = int(context.cache_seqlens[0].item()) if context.cache_seqlens is not None else 0
                total_len = chunk_start + q.shape[0]
                k_all = k_cache[0, :total_len]
                v_all = v_cache[0, :total_len]
                mask = self._offset_causal_mask(
                    q.shape[0],
                    total_len,
                    prefix_len=chunk_start,
                    device=q.device,
                    dtype=q.dtype,
                )
                return self._sdpa(q, k_all, v_all, attn_mask=mask)

            return self._sdpa(q, k, v, is_causal=True)

        if has_cache and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            total_len = int(context.context_lens.max().item()) if context.context_lens is not None else k_cache.shape[1]
            k_all = k_cache[0, :total_len]
            v_all = v_cache[0, :total_len]
            prefix_len = max(total_len - q.shape[0], 0)
            mask = self._offset_causal_mask(
                q.shape[0],
                total_len,
                prefix_len=prefix_len,
                device=q.device,
                dtype=q.dtype,
            )
            return self._sdpa(q, k_all, v_all, attn_mask=mask)

        return self._sdpa(q, k, v, is_causal=True)

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

        if not (HAS_FLASH_ATTN and q.is_cuda):
            return self._forward_torch(q, k, v)

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
