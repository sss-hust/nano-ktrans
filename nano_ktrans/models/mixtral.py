"""
Generic MoE model backbone for nano-ktrans.
Supports Mixtral-style sparse MLPs and Qwen2-MoE style sparse layers with
shared experts, while keeping the original lightweight execution flow.
"""

from typing import List
import torch
from torch import nn

from nano_ktrans.layers.norm import RMSNorm
from nano_ktrans.layers.attention import Attention
from nano_ktrans.layers.linear import QKVParallelLinear, RowParallelLinear
from nano_ktrans.layers.rotary_embedding import get_rope
from nano_ktrans.layers.hybrid_moe import HybridMoE
from nano_ktrans.models.config import GenericMoeConfig

MixtralConfig = GenericMoeConfig


class SparseExpertMLP(nn.Module):
    def __init__(self, config: GenericMoeConfig, intermediate_size: int):
        super().__init__()
        self.ffn_dim = intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up
        self.hidden_act = config.hidden_act

    def forward(self, hidden_states):
        act_fn = getattr(nn.functional, self.hidden_act)
        current_hidden_states = act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralBlockSparseTop2MLP(SparseExpertMLP):
    """兼容旧测试和旧接口名。"""

    def __init__(self, config: GenericMoeConfig):
        super().__init__(config, config.intermediate_size)


class PackedSparseExpertMLP(nn.Module):
    def __init__(self, config: GenericMoeConfig, intermediate_size: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(2 * self.intermediate_size, self.hidden_dim)
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.hidden_act = config.hidden_act

    def forward(self, hidden_states):
        act_fn = getattr(nn.functional, self.hidden_act)
        gate, up = nn.functional.linear(hidden_states, self.gate_up_proj).chunk(2, dim=-1)
        current_hidden_states = act_fn(gate) * up
        return self.down_proj(current_hidden_states)


class MixtralAttention(nn.Module):
    def __init__(self, config: GenericMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.qkv_bias
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
        if config.arch.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

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

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        q, k = self.rotary_emb(positions, q, k)
        
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class MixtralDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: GenericMoeConfig, 
        layer_idx: int,
        gpu_experts_mask: torch.Tensor,
        weight_path: str = "",
        offload_backend: str = "cpu",
        offload_backend_kwargs: dict | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MixtralAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_moe_layer = config.is_moe_layer(layer_idx)

        if self.is_moe_layer:
            self.gate = nn.Linear(self.hidden_size, config.num_local_experts, bias=False)

            gpu_experts = nn.ModuleDict()
            for i in range(config.num_local_experts):
                if gpu_experts_mask is not None and gpu_experts_mask[i]:
                    if config.arch.experts_are_packed:
                        gpu_experts[str(i)] = PackedSparseExpertMLP(config, config.moe_intermediate_size)
                    else:
                        gpu_experts[str(i)] = SparseExpertMLP(config, config.moe_intermediate_size)

            self.hybrid_moe = HybridMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                gpu_experts=gpu_experts,
                gpu_experts_mask=gpu_experts_mask,
                layer_idx=layer_idx,
                weight_path=weight_path,
                offload_backend=offload_backend,
                offload_backend_kwargs=offload_backend_kwargs,
                router_use_softmax=config.arch.router_use_softmax,
                normalize_topk_prob=config.normalize_topk_prob,
                expert_key_template=config.arch.expert_key_template,
                expert_proj_names=config.arch.expert_proj_names,
            )

            if config.arch.has_shared_expert and config.shared_expert_intermediate_size:
                self.shared_expert = SparseExpertMLP(config, config.shared_expert_intermediate_size)
                self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
            else:
                self.shared_expert = None
                self.shared_expert_gate = None
        else:
            self.gate = None
            self.hybrid_moe = None
            self.shared_expert = None
            self.shared_expert_gate = None
            self.mlp = SparseExpertMLP(config, config.intermediate_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states: [total_tokens, hidden_size]  扁平化的 2D 张量
        """
        # 1. Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # 2. MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_input = hidden_states

        if self.is_moe_layer:
            router_logits = self.gate(mlp_input)
            hidden_states = self.hybrid_moe(mlp_input, router_logits)

            if self.shared_expert is not None and self.shared_expert_gate is not None:
                shared_output = self.shared_expert(mlp_input)
                shared_gate = torch.sigmoid(self.shared_expert_gate(mlp_input))
                hidden_states = hidden_states + (shared_gate * shared_output)
        else:
            hidden_states = self.mlp(mlp_input)
        
        hidden_states = residual + hidden_states
        return hidden_states


class MixtralModel(nn.Module):
    """
    Minimal Mixtral model body.
    """
    def __init__(
        self,
        config: GenericMoeConfig,
        layer_gpu_expert_masks: List[torch.Tensor],
        weight_path: str = "",
        offload_backend: str = "cpu",
        offload_backend_kwargs: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(
                config, 
                layer_idx,
                gpu_experts_mask=layer_gpu_expert_masks[layer_idx],
                weight_path=weight_path,
                offload_backend=offload_backend,
                offload_backend_kwargs=offload_backend_kwargs,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
    ):
        hidden_states = self.embed_tokens(input_ids)           # [batch, seq_len, hidden]
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.config.hidden_size)  # flatten to 2D

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(positions, hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, -1)     # restore to 3D
        return hidden_states


class MixtralForCausalLM(nn.Module):
    """
    Minimal Mixtral Language Model.
    """
    def __init__(
        self,
        config: GenericMoeConfig,
        layer_gpu_expert_masks: List[torch.Tensor],
        weight_path: str = "",
        offload_backend: str = "cpu",
        offload_backend_kwargs: dict | None = None,
    ):
        super().__init__()
        self.model = MixtralModel(
            config,
            layer_gpu_expert_masks,
            weight_path,
            offload_backend=offload_backend,
            offload_backend_kwargs=offload_backend_kwargs,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.packed_modules_mapping = {
            "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
            "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
            "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
            config.arch.router_prefix: ("gate", None),
            config.arch.experts_prefix: ("hybrid_moe.gpu_experts", None),
        }
        if config.arch.use_qk_norm:
            self.packed_modules_mapping["self_attn.q_norm"] = ("self_attn.q_norm", None)
            self.packed_modules_mapping["self_attn.k_norm"] = ("self_attn.k_norm", None)
        if config.arch.shared_expert_prefix:
            self.packed_modules_mapping[config.arch.shared_expert_prefix] = ("shared_expert", None)
        if config.arch.shared_expert_gate_prefix:
            self.packed_modules_mapping[config.arch.shared_expert_gate_prefix] = ("shared_expert_gate", None)
        self.weight_name_substitutions = [
            (".gate_proj.", ".w1."),
            (".down_proj.", ".w2."),
            (".up_proj.", ".w3."),
        ]
        if config.arch.experts_are_packed:
            self.weight_name_substitutions = [
                pair for pair in self.weight_name_substitutions if pair[0] != ".gate_proj." and pair[0] != ".up_proj."
            ]

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
    ):
        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits


GenericMoeForCausalLM = MixtralForCausalLM
