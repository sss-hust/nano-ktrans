from __future__ import annotations

import os
from glob import glob
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArchitectureSpec:
    name: str
    expert_key_template: str
    expert_proj_names: dict[str, str]
    router_prefix: str
    experts_prefix: str
    router_use_softmax: bool = False
    normalize_topk_prob: bool = True
    has_shared_expert: bool = False
    shared_expert_key_template: Optional[str] = None
    shared_expert_prefix: Optional[str] = None
    shared_expert_gate_prefix: Optional[str] = None
    use_qk_norm: bool = False
    experts_are_packed: bool = False


MIXTRAL_SPEC = ModelArchitectureSpec(
    name="mixtral",
    expert_key_template="model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
    expert_proj_names={"gate": "w1", "up": "w3", "down": "w2"},
    router_prefix="block_sparse_moe.gate",
    experts_prefix="block_sparse_moe.experts",
    router_use_softmax=False,
    normalize_topk_prob=True,
)


QWEN2_MOE_SPEC = ModelArchitectureSpec(
    name="qwen2_moe",
    expert_key_template="model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight",
    expert_proj_names={"gate": "gate", "up": "up", "down": "down"},
    router_prefix="mlp.gate",
    experts_prefix="mlp.experts",
    router_use_softmax=True,
    normalize_topk_prob=False,
    has_shared_expert=True,
    shared_expert_key_template="model.layers.{layer}.mlp.shared_expert.{proj}_proj.weight",
    shared_expert_prefix="mlp.shared_expert",
    shared_expert_gate_prefix="mlp.shared_expert_gate",
)


QWEN3_MOE_SPEC = ModelArchitectureSpec(
    name="qwen3_moe",
    expert_key_template="model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
    expert_proj_names={"gate_up": "gate_up_proj", "down": "down_proj"},
    router_prefix="mlp.gate",
    experts_prefix="mlp.experts",
    router_use_softmax=True,
    normalize_topk_prob=False,
    use_qk_norm=True,
    experts_are_packed=True,
)


QWEN3_MOE_UNPACKED_SPEC = ModelArchitectureSpec(
    name="qwen3_moe",
    expert_key_template="model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
    expert_proj_names={"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
    router_prefix="mlp.gate",
    experts_prefix="mlp.experts",
    router_use_softmax=True,
    normalize_topk_prob=False,
    use_qk_norm=True,
    experts_are_packed=False,
)


class GenericMoeConfig:
    def __init__(
        self,
        *,
        arch: ModelArchitectureSpec = MIXTRAL_SPEC,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        moe_intermediate_size: Optional[int] = None,
        shared_expert_intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        hidden_act: str = "silu",
        qkv_bias: bool = False,
        head_dim: Optional[int] = None,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Optional[list[int]] = None,
        first_k_dense_replace: int = 0,
        attention_backend: str = "standard",
        normalize_topk_prob: Optional[bool] = None,
    ):
        self.arch = arch
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size or intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = set(mlp_only_layers or [])
        self.first_k_dense_replace = first_k_dense_replace
        self.attention_backend = attention_backend
        self.normalize_topk_prob = arch.normalize_topk_prob if normalize_topk_prob is None else normalize_topk_prob

    @classmethod
    def from_hf_config(cls, hf_config) -> "GenericMoeConfig":
        model_type = getattr(hf_config, "model_type", "")
        arch = infer_architecture(hf_config)
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
            rope_theta = rope_parameters.get("rope_theta", 1000000.0)

        if model_type.startswith("deepseek"):
            attention_backend = "mla"
        else:
            attention_backend = "standard"

        num_local_experts = (
            getattr(hf_config, "num_local_experts", None)
            or getattr(hf_config, "num_experts", None)
            or getattr(hf_config, "n_routed_experts", None)
            or 0
        )

        num_key_value_heads = getattr(hf_config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = getattr(hf_config, "num_attention_heads")

        return cls(
            arch=arch,
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4),
            moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", None),
            shared_expert_intermediate_size=getattr(hf_config, "shared_expert_intermediate_size", None),
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_local_experts=num_local_experts,
            num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 2) or 0,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 32768),
            rope_theta=rope_theta,
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            qkv_bias=getattr(hf_config, "qkv_bias", getattr(hf_config, "attention_bias", False)),
            head_dim=getattr(hf_config, "head_dim", None),
            decoder_sparse_step=getattr(hf_config, "decoder_sparse_step", 1) or 1,
            mlp_only_layers=list(getattr(hf_config, "mlp_only_layers", []) or []),
            first_k_dense_replace=getattr(hf_config, "first_k_dense_replace", 0) or 0,
            attention_backend=attention_backend,
            normalize_topk_prob=getattr(hf_config, "norm_topk_prob", None),
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.attention_backend != "standard":
            return False

        if self.num_local_experts <= 0 or self.num_experts_per_tok <= 0:
            return False

        if layer_idx < self.first_k_dense_replace:
            return False

        if layer_idx in self.mlp_only_layers:
            return False

        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @property
    def supports_cpu_offload(self) -> bool:
        return self.attention_backend == "standard" and self.num_local_experts > 0


def infer_architecture(hf_config) -> ModelArchitectureSpec:
    model_type = getattr(hf_config, "model_type", "")
    architectures = [a.lower() for a in getattr(hf_config, "architectures", []) or []]

    if model_type == "mixtral":
        return MIXTRAL_SPEC

    if model_type == "qwen2_moe" or any("qwen2moe" in a for a in architectures):
        return QWEN2_MOE_SPEC

    if model_type == "qwen3_moe" or any("qwen3moe" in a for a in architectures):
        return QWEN3_MOE_SPEC

    if getattr(hf_config, "shared_expert_intermediate_size", None) is not None:
        return QWEN2_MOE_SPEC

    return MIXTRAL_SPEC


def adapt_config_to_checkpoint(config: GenericMoeConfig, checkpoint_path: str) -> GenericMoeConfig:
    """
    Some Qwen3-MoE checkpoints store expert projections as separate gate/up/down weights
    instead of a packed gate_up tensor. Detect the real layout from safetensor keys so the
    Python model and the offload loader agree on parameter names.
    """
    if config.arch.name != "qwen3_moe":
        return config

    packed_key = "model.layers.0.mlp.experts.0.gate_up_proj.weight"
    unpacked_key = "model.layers.0.mlp.experts.0.gate_proj.weight"

    try:
        from safetensors import safe_open
    except ImportError:
        return config

    for file_path in sorted(glob(os.path.join(checkpoint_path, "*.safetensors"))):
        with safe_open(file_path, "pt", "cpu") as handle:
            keys = handle.keys()
            if packed_key in keys:
                return config
            if unpacked_key in keys:
                config.arch = QWEN3_MOE_UNPACKED_SPEC
                return config

    return config
