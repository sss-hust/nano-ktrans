import torch
from nano_ktrans.utils.context import set_context, reset_context

class SimpleEngine:
    """
    A minimal execution engine for text generation without complex batching.
    Supports a single sequence at a time (batch size = 1).
    """
    def __init__(self, model, max_seq_len=2048):
        self.model = model
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Pre-allocate contiguous KV cache for batch size = 1
        config = self.model.model.config
        
        self.k_caches = []
        self.v_caches = []
        
        head_dim = config.hidden_size // config.num_attention_heads
        
        for layer in self.model.model.layers:
            # Shape for flash_attn_with_kvcache with block_table=None is:
            # [batch_size, seq_len, num_kv_heads, head_dim]
            k_cache = torch.zeros(
                1, max_seq_len, config.num_key_value_heads, head_dim, 
                device=self.device, dtype=self.dtype
            )
            v_cache = torch.zeros_like(k_cache)
            
            # Mount cache to the attention layer
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)
            
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Process the initial prompt.
        """
        seq_len = input_ids.shape[1]
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported in SimpleEngine."
        assert seq_len <= self.max_seq_len, f"Prompt length ({seq_len}) exceeds max sequence length ({self.max_seq_len})."
        
        # `positions` for Rotary Embedding
        positions = torch.arange(seq_len, device=self.device)
        
        # slot_mapping for KV Cache (contiguous indices for batch 0)
        # Since Triton kernel `store_kvcache` expects a flat mapping, and our 
        # k_cache layout is [1, seq_len, num_kv_heads, head_dim],
        # the slot index corresponding to seq_idx is exactly seq_idx.
        slot_mapping = positions.clone().to(torch.int32)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = cu_seqlens_q
        
        # Set Global Context
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
        )
        
        logits = self.model(input_ids, positions)
        reset_context()
        return logits

    @torch.no_grad()
    def decode_step(self, input_id: torch.Tensor, current_seq_len: int) -> torch.Tensor:
        """
        Generate a single token.
        """
        assert input_id.shape == (1, 1), "Decode step requires input shape (1, 1)"
        assert current_seq_len < self.max_seq_len, "KV Cache exhausted."
        
        positions = torch.tensor([current_seq_len], device=self.device)
        slot_mapping = positions.clone().to(torch.int32)
        
        # flash_attn_with_kvcache expects context_lens for each sequence in the batch
        context_lens = torch.tensor([current_seq_len + 1], dtype=torch.int32, device=self.device)
        
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=None,
        )
        
        logits = self.model(input_id, positions)
        reset_context()
        return logits
