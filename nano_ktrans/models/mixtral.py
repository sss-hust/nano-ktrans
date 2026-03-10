"""
Simplified Mixtral-8x7B MoE model for nano-ktrans.
This implementation focuses purely on the single-node forward pass, integrating
our Triton-based Attention and CPU/GPU Hybrid MoE layers.
"""

import math
from typing import Optional, Tuple, List
import torch
from torch import nn

from nano_ktrans.layers.norm import RMSNorm
from nano_ktrans.layers.attention import Attention
from nano_ktrans.layers.linear import QKVParallelLinear, RowParallelLinear
from nano_ktrans.layers.rotary_embedding import get_rope
from nano_ktrans.layers.hybrid_moe import HybridMoE

class MixtralConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

class MixtralBlockSparseTop2MLP(nn.Module):
    """A single Mixtral expert."""
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up
        self.act_fn = nn.functional.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralAttention(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        scaling = self.head_dim ** -0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        q, k = self.rotary_emb(positions, q, k)
        
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class MixtralDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: MixtralConfig, 
        layer_idx: int,
        gpu_experts_mask: torch.Tensor,
        weight_path: str = ""
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MixtralAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 1. Gate is executed eagerly on the main GPU
        self.gate = nn.Linear(self.hidden_size, config.num_local_experts, bias=False)
        
        # 2. Local GPU experts
        gpu_experts = nn.ModuleDict()
        for i in range(config.num_local_experts):
            if gpu_experts_mask is not None and gpu_experts_mask[i]:
                gpu_experts[str(i)] = MixtralBlockSparseTop2MLP(config)
        
        # 3. Hybrid MoE orchestrator
        self.hybrid_moe = HybridMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.intermediate_size,
            gpu_experts=gpu_experts,
            gpu_experts_mask=gpu_experts_mask,
            layer_idx=layer_idx,
            weight_path=weight_path,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        
        # 1. Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # 2. MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        batch_seq_len, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Route
        router_logits = self.gate(flat_hidden_states)
        
        # Execute Hybrid MoE (CPU + GPU concurrent)
        hidden_states = self.hybrid_moe(flat_hidden_states, router_logits)
        hidden_states = hidden_states.view(batch_seq_len, hidden_dim)
        
        hidden_states = residual + hidden_states
        return hidden_states


class MixtralModel(nn.Module):
    """
    Minimal Mixtral model body.
    """
    def __init__(self, config: MixtralConfig, layer_gpu_expert_masks: List[torch.Tensor], weight_path: str = ""):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(
                config, 
                layer_idx,
                gpu_experts_mask=layer_gpu_expert_masks[layer_idx],
                weight_path=weight_path
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
    ):
        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(positions, hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    """
    Minimal Mixtral Language Model.
    """
    packed_modules_mapping = {
        "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
        "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
        "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
        "block_sparse_moe.gate": ("gate", None),
        "block_sparse_moe.experts": ("hybrid_moe.gpu_experts", None),
    }

    def __init__(self, config: MixtralConfig, layer_gpu_expert_masks: List[torch.Tensor], weight_path: str = ""):
        super().__init__()
        self.model = MixtralModel(config, layer_gpu_expert_masks, weight_path)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
    ):
        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits
