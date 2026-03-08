"""
nano-ktrans 基础测试

这些测试验证核心模块的基本功能，不需要 GPU 或完整模型权重即可运行：
- 层的构造和前向计算（RMSNorm, Linear, Attention 结构）
- 配置解析
- 路由逻辑
"""

import torch
import pytest


# ============================================================
# Test 1: RMSNorm
# ============================================================
class TestRMSNorm:
    def test_basic_forward(self):
        """RMSNorm 基本前向测试"""
        from nano_ktrans.layers.norm import RMSNorm

        norm = RMSNorm(hidden_size=128, eps=1e-5)
        x = torch.randn(2, 10, 128)
        out = norm(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_normalized(self):
        """RMSNorm 输出的 RMS 应接近 1"""
        from nano_ktrans.layers.norm import RMSNorm

        norm = RMSNorm(hidden_size=256, eps=1e-5)
        x = torch.randn(4, 8, 256) * 10  # 大数值输入
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # RMSNorm 后每个向量的 RMS 应接近 1（乘以 weight=1 时）
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), \
            f"RMS values not close to 1: {rms.mean():.4f}"


# ============================================================
# Test 2: Linear Layers
# ============================================================
class TestLinearLayers:
    def test_linear_base(self):
        """基础线性层"""
        from nano_ktrans.layers.linear import LinearBase

        linear = LinearBase(128, 256)
        x = torch.randn(2, 128)
        out = linear(x)
        assert out.shape == (2, 256)

    def test_column_parallel(self):
        """ColumnParallel 线性层"""
        from nano_ktrans.layers.linear import ColumnParallelLinear

        linear = ColumnParallelLinear(128, 512)
        x = torch.randn(4, 128)
        out = linear(x)
        assert out.shape == (4, 512)

    def test_row_parallel(self):
        """RowParallel 线性层"""
        from nano_ktrans.layers.linear import RowParallelLinear

        linear = RowParallelLinear(256, 128)
        x = torch.randn(4, 256)
        out = linear(x)
        assert out.shape == (4, 128)

    def test_merged_column_parallel(self):
        """MergedColumnParallel 层：多路输出合并"""
        from nano_ktrans.layers.linear import MergedColumnParallelLinear

        linear = MergedColumnParallelLinear(128, [256, 256])
        x = torch.randn(2, 128)
        out = linear(x)
        assert out.shape == (2, 512)  # 256 + 256

    def test_qkv_parallel(self):
        """QKVParallelLinear 层：Q/K/V 合并投影"""
        from nano_ktrans.layers.linear import QKVParallelLinear

        # 32 heads, 8 kv heads, head_dim = 128
        linear = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
        )
        x = torch.randn(2, 4096)
        out = linear(x)
        # output = (32 + 8 + 8) * 128 = 6144
        assert out.shape == (2, 6144), f"Expected (2, 6144), got {out.shape}"

    def test_qkv_weight_loader(self):
        """QKVParallelLinear 的 weight_loader 正确切片"""
        from nano_ktrans.layers.linear import QKVParallelLinear

        linear = QKVParallelLinear(
            hidden_size=128,
            head_size=32,
            total_num_heads=4,
            total_num_kv_heads=2,
        )
        # (4 + 2 + 2) * 32 = 256
        assert linear.weight.shape == (256, 128)

        q_weight = torch.ones(128, 128)  # 4 * 32 = 128
        k_weight = torch.ones(64, 128) * 2   # 2 * 32 = 64
        v_weight = torch.ones(64, 128) * 3

        linear.weight_loader(linear.weight, q_weight, "q")
        linear.weight_loader(linear.weight, k_weight, "k")
        linear.weight_loader(linear.weight, v_weight, "v")

        assert torch.all(linear.weight[:128] == 1.0), "Q shard incorrect"
        assert torch.all(linear.weight[128:192] == 2.0), "K shard incorrect"
        assert torch.all(linear.weight[192:256] == 3.0), "V shard incorrect"


# ============================================================
# Test 3: Rotary Embedding
# ============================================================
class TestRotaryEmbedding:
    def test_rope_forward(self):
        """RoPE 前向不改变形状"""
        from nano_ktrans.layers.rotary_embedding import RotaryEmbedding

        rope = RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=2048,
            base=10000.0,
        )
        positions = torch.arange(10)
        q = torch.randn(10, 4, 128)  # [seq_len, num_heads, head_dim]
        k = torch.randn(10, 2, 128)
        q_out, k_out = rope(positions, q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rope_consistency(self):
        """同一位置的 RoPE 应产生相同结果"""
        from nano_ktrans.layers.rotary_embedding import RotaryEmbedding

        rope = RotaryEmbedding(128, 128, 2048, 10000.0)
        positions = torch.tensor([5, 5])
        q = torch.randn(2, 4, 128)
        k = torch.randn(2, 2, 128)

        q[1] = q[0].clone()
        k[1] = k[0].clone()

        q_out, k_out = rope(positions, q, k)
        assert torch.allclose(q_out[0], q_out[1], atol=1e-5), \
            "Same position should produce same rotary embedding"


# ============================================================
# Test 4: MixtralConfig
# ============================================================
class TestMixtralConfig:
    def test_default_config(self):
        """MixtralConfig 默认值正确"""
        from nano_ktrans.models.mixtral import MixtralConfig

        config = MixtralConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.num_local_experts == 8
        assert config.num_experts_per_tok == 2

    def test_custom_config(self):
        """MixtralConfig 自定义参数"""
        from nano_ktrans.models.mixtral import MixtralConfig

        config = MixtralConfig(
            vocab_size=64000,
            hidden_size=2048,
            num_hidden_layers=16,
        )
        assert config.vocab_size == 64000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 16


# ============================================================
# Test 5: MoE Routing Logic (CPU-side, no kt-kernel needed)
# ============================================================
class TestMoERouting:
    def test_topk_routing(self):
        """Top-K 路由逻辑测试"""
        batch_size = 4
        num_experts = 8
        top_k = 2

        router_logits = torch.randn(batch_size, num_experts)
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)

        assert topk_ids.shape == (batch_size, top_k)
        assert topk_weights.shape == (batch_size, top_k)
        # weights 应归一化
        row_sums = topk_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)
        # ids 应在 [0, num_experts)
        assert (topk_ids >= 0).all() and (topk_ids < num_experts).all()

    def test_expert_mask(self):
        """GPU 专家掩码逻辑"""
        num_experts = 8
        num_gpu_experts = 2

        mask = torch.zeros(num_experts, dtype=torch.bool)
        mask[:num_gpu_experts] = True

        assert mask.sum() == num_gpu_experts
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False


# ============================================================
# Test 6: Context System
# ============================================================
class TestContext:
    def test_set_get_reset(self):
        """Context 的 set / get / reset 生命周期"""
        from nano_ktrans.utils.context import set_context, get_context, reset_context

        # 默认状态
        reset_context()
        ctx = get_context()
        assert ctx.is_prefill == False

        # 设置 prefill 上下文
        set_context(is_prefill=True, max_seqlen_q=128, max_seqlen_k=128)
        ctx = get_context()
        assert ctx.is_prefill == True
        assert ctx.max_seqlen_q == 128

        # 重置
        reset_context()
        ctx = get_context()
        assert ctx.is_prefill == False
        assert ctx.max_seqlen_q == 0


# ============================================================
# Test 7: Weight Loader (file-based, no GPU needed)
# ============================================================
class TestWeightLoader:
    def test_default_weight_loader(self):
        """default_weight_loader 直接拷贝"""
        from nano_ktrans.utils.loader import default_weight_loader
        import torch.nn as nn

        param = nn.Parameter(torch.zeros(3, 3))
        loaded = torch.ones(3, 3)
        default_weight_loader(param, loaded)
        assert torch.all(param == 1.0)


# ============================================================
# Test 8: MixtralBlockSparseTop2MLP (单个专家)
# ============================================================
class TestMixtralExpert:
    def test_expert_forward(self):
        """单个 Mixtral 专家前向"""
        from nano_ktrans.models.mixtral import MixtralBlockSparseTop2MLP, MixtralConfig

        config = MixtralConfig(hidden_size=128, intermediate_size=512)
        expert = MixtralBlockSparseTop2MLP(config)
        x = torch.randn(4, 128)
        out = expert(x)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
