import torch
from nano_ktrans.utils.context import set_context, reset_context

class SimpleEngine:
    """
    A minimal execution engine for text generation without complex batching.
    Supports a single sequence at a time (batch size = 1).

    支持两种 prefill 模式：
    - 标准 prefill：一次性处理整个序列（flash_attn_varlen_func）
    - Chunked prefill：将长序列分块处理，每块通过完整模型，降低显存峰值
      （使用 flash_attn_with_kvcache 的 k=/v= 追加模式）
    """
    def __init__(self, model, max_seq_len=2048, chunk_size=512):
        self.model = model
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Pre-allocate contiguous KV cache for batch size = 1
        config = self.model.model.config
        
        self.k_caches = []
        self.v_caches = []
        
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        
        for layer in self.model.model.layers:
            # Shape for flash_attn_with_kvcache with block_table=None is:
            # [batch_size, seq_len, num_kv_heads, head_dim]
            k_cache = torch.zeros(
                1, max_seq_len, config.num_key_value_heads, head_dim, 
                device=self.device, dtype=self.dtype
            )
            v_cache = torch.zeros_like(k_cache)
            
            # 修复: 将 cache 挂载到 Attention 实例而非 MixtralAttention 实例
            layer.self_attn.attn.k_cache = k_cache
            layer.self_attn.attn.v_cache = v_cache
            
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)

    def _refresh_offload_state(self, *, phase: str = "decode") -> int:
        background_fn = getattr(self.model.model, "background_tick_offload_state", None)
        if background_fn is not None:
            background_fn(phase=phase)
        refresh_fn = getattr(self.model.model, "refresh_offload_state", None)
        if refresh_fn is None:
            return 0
        return int(refresh_fn(phase=phase))

    def start_background_offload_worker(self) -> bool:
        start_fn = getattr(self.model.model, "start_offload_worker", None)
        if start_fn is None:
            return False
        start_fn()
        return True

    def stop_background_offload_worker(self) -> bool:
        stop_fn = getattr(self.model.model, "shutdown_offload_worker", None)
        if stop_fn is None:
            return False
        stop_fn()
        return True
            
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        处理初始 prompt。长序列自动切换到 chunked prefill。
        """
        seq_len = input_ids.shape[1]
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported."
        assert seq_len <= self.max_seq_len, f"Prompt ({seq_len}) > max_seq_len ({self.max_seq_len})."
        
        if seq_len > self.chunk_size:
            return self._prefill_chunked(input_ids)
        return self._prefill_full(input_ids)

    @torch.no_grad()
    def _prefill_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """标准全量 prefill（短序列，一次处理完）。"""
        self._refresh_offload_state(phase="prefill")
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=self.device)
        slot_mapping = positions.clone().to(torch.int32)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = cu_seqlens_q
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
        )
        
        logits = self.model(input_ids, positions)
        reset_context()
        return logits

    @torch.no_grad()
    def _prefill_chunked(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        分块 prefill（长序列）。

        将序列切分为 chunk_size 大小的块，每块完整通过模型的所有层。
        每块的注意力通过 flash_attn_with_kvcache 的 k=/v= 追加模式，
        既写入 KV Cache 又对所有已缓存 token 做因果注意力。
        """
        seq_len = input_ids.shape[1]
        logits = None
        
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            
            self._refresh_offload_state(phase="prefill")
            positions = torch.arange(chunk_start, chunk_end, device=self.device)
            cache_seqlens = torch.tensor([chunk_start], dtype=torch.int32, device=self.device)
            
            set_context(
                is_prefill=True,
                is_chunked_prefill=True,
                cache_seqlens=cache_seqlens,
            )
            
            logits = self.model(chunk_ids, positions)
            reset_context()
        
        return logits

    @torch.no_grad()
    def decode_step(self, input_id: torch.Tensor, current_seq_len: int) -> torch.Tensor:
        """Generate a single token."""
        self._refresh_offload_state(phase="decode")
        assert input_id.shape == (1, 1), "Decode step requires input shape (1, 1)"
        assert current_seq_len < self.max_seq_len, "KV Cache exhausted."
        
        positions = torch.tensor([current_seq_len], device=self.device)
        slot_mapping = positions.clone().to(torch.int32)
        
        # context_lens 表示 cache 中有效 token 数（包含当前 token）
        context_lens = torch.tensor([current_seq_len + 1], dtype=torch.int32, device=self.device)
        
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
        )
        
        logits = self.model(input_id, positions)
        reset_context()
        return logits
