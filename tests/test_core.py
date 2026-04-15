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
        assert config.arch.name == "mixtral"

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


class TestArchitectureConfig:
    def test_qwen2_moe_config_mapping(self):
        from types import SimpleNamespace
        from nano_ktrans.models.config import GenericMoeConfig

        hf_config = SimpleNamespace(
            model_type="qwen2_moe",
            architectures=["Qwen2MoeForCausalLM"],
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=5632,
            moe_intermediate_size=1408,
            shared_expert_intermediate_size=5632,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            num_experts=60,
            num_experts_per_tok=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            rope_theta=10000.0,
            hidden_act="silu",
            decoder_sparse_step=2,
            mlp_only_layers=[0, 1],
            qkv_bias=True,
        )

        config = GenericMoeConfig.from_hf_config(hf_config)
        assert config.arch.name == "qwen2_moe"
        assert config.num_local_experts == 60
        assert config.moe_intermediate_size == 1408
        assert config.shared_expert_intermediate_size == 5632
        assert config.is_moe_layer(0) is False
        assert config.is_moe_layer(1) is False
        assert config.is_moe_layer(2) is False
        assert config.is_moe_layer(3) is True

    def test_deepseek_marks_attention_gap(self):
        from types import SimpleNamespace
        from nano_ktrans.models.config import GenericMoeConfig

        hf_config = SimpleNamespace(
            model_type="deepseek_v2",
            architectures=["DeepseekV2ForCausalLM"],
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            moe_intermediate_size=1407,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            n_routed_experts=64,
            num_experts_per_tok=6,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            hidden_act="silu",
            attention_bias=False,
        )

        config = GenericMoeConfig.from_hf_config(hf_config)
        assert config.attention_backend == "mla"

    def test_qwen3_moe_config_mapping(self):
        from types import SimpleNamespace
        from nano_ktrans.models.config import GenericMoeConfig

        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_experts=128,
            num_experts_per_tok=8,
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            rope_parameters={"rope_theta": 10000.0},
            hidden_act="silu",
            decoder_sparse_step=1,
            mlp_only_layers=[],
            attention_bias=False,
            head_dim=128,
        )

        config = GenericMoeConfig.from_hf_config(hf_config)
        assert config.arch.name == "qwen3_moe"
        assert config.arch.use_qk_norm is True
        assert config.arch.experts_are_packed is True
        assert config.num_local_experts == 128
        assert config.num_key_value_heads == 4

    def test_qwen3_checkpoint_layout_adaptation(self, tmp_path):
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from nano_ktrans.models.config import GenericMoeConfig, adapt_config_to_checkpoint

        hf_config = SimpleNamespace(
            model_type="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
            vocab_size=151936,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            max_position_embeddings=1024,
            rope_parameters={"rope_theta": 10000.0},
            hidden_act="silu",
            decoder_sparse_step=1,
            mlp_only_layers=[],
            attention_bias=False,
            head_dim=32,
        )

        save_file(
            {
                "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(64, 128),
                "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(64, 128),
                "model.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(128, 64),
            },
            str(tmp_path / "model.safetensors"),
        )

        config = GenericMoeConfig.from_hf_config(hf_config)
        assert config.arch.experts_are_packed is True

        config = adapt_config_to_checkpoint(config, str(tmp_path))
        assert config.arch.experts_are_packed is False
        assert config.arch.expert_proj_names == {
            "gate": "gate_proj",
            "up": "up_proj",
            "down": "down_proj",
        }


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

    def test_chunked_prefill_context(self):
        """Chunked prefill 上下文字段"""
        from nano_ktrans.utils.context import set_context, get_context, reset_context

        cache_seqlens = torch.tensor([64], dtype=torch.int32)
        set_context(
            is_prefill=True,
            is_chunked_prefill=True,
            cache_seqlens=cache_seqlens,
        )
        ctx = get_context()
        assert ctx.is_prefill == True
        assert ctx.is_chunked_prefill == True
        assert torch.equal(ctx.cache_seqlens, cache_seqlens)
        reset_context()
        ctx = get_context()
        assert ctx.is_chunked_prefill == False
        assert ctx.cache_seqlens is None


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


# ============================================================
# Test 9: GPU Expert Selection (generate_gpu_experts_masks)
# ============================================================
class TestExpertSelection:
    def test_generate_masks_basic(self):
        """基于激活频率选择 GPU 专家"""
        from nano_ktrans.utils.expert_selection import generate_gpu_experts_masks

        # 2 层，8 个专家
        freq = torch.tensor([
            [10, 2, 8, 1, 5, 3, 7, 4],  # layer 0: expert 0 和 2 最活跃
            [1, 9, 3, 8, 2, 6, 4, 7],   # layer 1: expert 1 和 3 最活跃
        ], dtype=torch.float)

        masks = generate_gpu_experts_masks(freq, num_gpu_experts=2)
        assert len(masks) == 2

        # layer 0: experts 0 (freq=10) 和 2 (freq=8) 应在 GPU
        assert masks[0][0] == True and masks[0][2] == True
        assert masks[0].sum() == 2

        # layer 1: experts 1 (freq=9) 和 3 (freq=8) 应在 GPU
        assert masks[1][1] == True and masks[1][3] == True
        assert masks[1].sum() == 2

    def test_generate_masks_all_gpu(self):
        """num_gpu_experts >= num_experts 时所有专家都在 GPU"""
        from nano_ktrans.utils.expert_selection import generate_gpu_experts_masks

        freq = torch.rand(4, 8)
        masks = generate_gpu_experts_masks(freq, num_gpu_experts=10)
        for mask in masks:
            assert mask.all(), "All experts should be on GPU"

    def test_uniform_masks(self):
        """均匀选择 fallback"""
        from nano_ktrans.utils.expert_selection import uniform_gpu_experts_masks

        masks = uniform_gpu_experts_masks(num_layers=4, num_experts=8, num_gpu_experts=3)
        assert len(masks) == 4
        for mask in masks:
            assert mask.sum() == 3
            assert mask[0] == True and mask[1] == True and mask[2] == True
            assert mask[3] == False

    def test_different_layers_different_experts(self):
        """不同层应该选择不同的热门专家"""
        from nano_ktrans.utils.expert_selection import generate_gpu_experts_masks

        freq = torch.tensor([
            [100, 1, 1, 1],   # layer 0: only expert 0 is hot
            [1, 100, 1, 1],   # layer 1: only expert 1 is hot
            [1, 1, 100, 1],   # layer 2: only expert 2 is hot
        ], dtype=torch.float)

        masks = generate_gpu_experts_masks(freq, num_gpu_experts=1)
        assert masks[0][0] == True and masks[0].sum() == 1
        assert masks[1][1] == True and masks[1].sum() == 1
        assert masks[2][2] == True and masks[2].sum() == 1


class TestOffloadBackendHelpers:
    def test_normalize_offload_backend_name(self):
        from nano_ktrans.kernels.offload_backend import normalize_offload_backend_name

        assert normalize_offload_backend_name(None) == "cpu"
        assert normalize_offload_backend_name("cpu") == "cpu"
        assert normalize_offload_backend_name("pim") == "pim"
        assert normalize_offload_backend_name("pim-shadow") == "pim_shadow"


class TestDynamicScheduler:
    def test_residency_plan_from_gpu_masks(self):
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        masks = [
            torch.tensor([True, False, False, True], dtype=torch.bool),
            torch.tensor([False, True, False, False], dtype=torch.bool),
        ]
        plan = ExpertResidencyPlan.from_gpu_masks(masks, default_offload_tier=ExpertResidency.PIM)
        summary = plan.summary()
        assert summary["layers"][0]["gpu_experts"] == 2
        assert summary["layers"][0]["pim_experts"] == 2
        assert summary["layers"][1]["gpu_experts"] == 1

    def test_scheduler_prefill_promotes_to_budget(self):
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        masks = [torch.tensor([True, False, False, False], dtype=torch.bool)]
        plan = ExpertResidencyPlan.from_gpu_masks(masks, default_offload_tier=ExpertResidency.PIM)
        scheduler = DynamicExpertScheduler(
            residency_plan=plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                prefill_force_gpu_budget_per_layer=2,
                offload_tier=ExpertResidency.PIM,
            ),
        )
        scheduler.observe(0, torch.tensor([[2, 3], [2, 2]]), phase="prefill")
        ops = scheduler.plan_layer(0, phase="prefill")
        promoted = [op for op in ops if op.dst == ExpertResidency.GPU]
        assert len(promoted) >= 1

    def test_migration_manager_records_phase(self):
        from nano_ktrans.kernels.offload_backend import ExpertOffloadBackend
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        class DummyBackend(ExpertOffloadBackend):
            def submit_forward(self, hidden_states, topk_ids, topk_weights, cuda_stream):
                return None

            def sync_forward(self, hidden_states, cuda_stream):
                return hidden_states

            def update_gpu_expert_mask(self, gpu_experts_mask):
                return None

        backend = DummyBackend()
        backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=3,
                    expert_idx=7,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="promote_hot_expert",
                )
            ],
            phase="prefill",
        )
        diagnostics = backend.diagnostics()
        assert diagnostics["migration_submit_calls"] == 1
        assert diagnostics["migration_manager"]["layers"][0]["history"][0]["phase"] == "prefill"

    def test_hybrid_moe_applies_decode_migration_plan(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        weight_path = tmp_path / "weights"
        weight_path.mkdir()
        save_file(
            {
                "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.0.w2.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.1.w2.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.randn(8, 4),
            },
            str(weight_path / "model.safetensors"),
        )

        gpu_mask = torch.tensor([True, False], dtype=torch.bool)
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            [gpu_mask],
            default_offload_tier=ExpertResidency.PIM,
        )
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )

        hybrid = HybridMoE(
            num_experts=2,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=8,
            gpu_experts=torch.nn.ModuleDict(),
            gpu_experts_mask=gpu_mask.clone(),
            layer_idx=0,
            weight_path=str(weight_path),
            offload_backend="cpu",
            residency_plan=residency_plan,
            dynamic_expert_scheduler=scheduler,
            hidden_act="silu",
        ).to(dtype=torch.float32)

        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="test_promote",
                )
            ],
            phase="decode",
        )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        assert "1" in hybrid.gpu_experts
        assert bool(hybrid.gpu_experts_mask[1].item()) is True
        assert bool(hybrid.gpu_experts_mask[0].item()) is False

        diagnostics = hybrid.diagnostics()
        assert diagnostics["applied_migration_ops"] == 2
        assert diagnostics["last_applied_migration_phase"] == "decode"
        assert diagnostics["gpu_experts_mask_sum"] == 1

    def test_hybrid_moe_prefetches_during_prefill(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        weight_path = tmp_path / "weights"
        weight_path.mkdir()
        save_file(
            {
                "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.0.w2.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(8, 4),
                "model.layers.0.block_sparse_moe.experts.1.w2.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.randn(8, 4),
            },
            str(weight_path / "model.safetensors"),
        )

        gpu_mask = torch.tensor([True, False], dtype=torch.bool)
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            [gpu_mask],
            default_offload_tier=ExpertResidency.PIM,
        )
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                prefill_force_gpu_budget_per_layer=2,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )

        hybrid = HybridMoE(
            num_experts=2,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=8,
            gpu_experts=torch.nn.ModuleDict(),
            gpu_experts_mask=gpu_mask.clone(),
            layer_idx=0,
            weight_path=str(weight_path),
            offload_backend="cpu",
            residency_plan=residency_plan,
            dynamic_expert_scheduler=scheduler,
            hidden_act="silu",
            expert_prefetch_workers=0,
        ).to(dtype=torch.float32)

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        set_context(is_prefill=True)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        diagnostics = hybrid.diagnostics()
        assert diagnostics["prefetch_requested"] >= 1
        assert diagnostics["materialization_manager"]["pending_prefetches"] == 0

    def test_hybrid_moe_eviction_for_promotion_respects_budget(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

        weight_path = tmp_path / "weights"
        weight_path.mkdir()
        tensors = {}
        for expert_idx in range(3):
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w1.weight"] = torch.randn(8, 4)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w2.weight"] = torch.randn(4, 8)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w3.weight"] = torch.randn(8, 4)
        save_file(tensors, str(weight_path / "model.safetensors"))

        gpu_mask = torch.tensor([True, False, False], dtype=torch.bool)
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            [gpu_mask],
            default_offload_tier=ExpertResidency.PIM,
        )
        residency_plan.layer_state(0).hotness = torch.tensor([0.1, 0.9, 0.8], dtype=torch.float32)
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )

        hybrid = HybridMoE(
            num_experts=3,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=8,
            gpu_experts=torch.nn.ModuleDict(),
            gpu_experts_mask=gpu_mask.clone(),
            layer_idx=0,
            weight_path=str(weight_path),
            offload_backend="cpu",
            residency_plan=residency_plan,
            dynamic_expert_scheduler=scheduler,
            hidden_act="silu",
            expert_prefetch_workers=0,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="promote_hot_expert",
                )
            ],
            phase="decode",
        )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        assert bool(hybrid.gpu_experts_mask[1].item()) is True
        assert bool(hybrid.gpu_experts_mask[0].item()) is False
        assert int(hybrid.gpu_experts_mask.bool().sum().item()) == 1

        diagnostics = hybrid.diagnostics()
        assert diagnostics["runtime_evictions"] == 1

    def test_hybrid_moe_decode_prefetches_future_promotions(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        weight_path = tmp_path / "weights"
        weight_path.mkdir()
        tensors = {}
        for expert_idx in range(4):
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w1.weight"] = torch.randn(8, 4)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w2.weight"] = torch.randn(4, 8)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w3.weight"] = torch.randn(8, 4)
        save_file(tensors, str(weight_path / "model.safetensors"))

        gpu_mask = torch.tensor([True, False, False, False], dtype=torch.bool)
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            [gpu_mask],
            default_offload_tier=ExpertResidency.PIM,
        )
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=2,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
            top_k=2,
            hidden_size=4,
            moe_intermediate_size=8,
            gpu_experts=torch.nn.ModuleDict(),
            gpu_experts_mask=gpu_mask.clone(),
            layer_idx=0,
            weight_path=str(weight_path),
            offload_backend="cpu",
            residency_plan=residency_plan,
            dynamic_expert_scheduler=scheduler,
            hidden_act="silu",
            expert_prefetch_workers=0,
        ).to(dtype=torch.float32)

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.1, 4.0, 3.0, 2.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        diagnostics = hybrid.diagnostics()
        assert diagnostics["prefetch_requested"] >= 2
        assert diagnostics["materialization_manager"]["prefetch_resolved"] >= 1
        assert diagnostics["materialization_manager"]["cache_size"] >= 1
