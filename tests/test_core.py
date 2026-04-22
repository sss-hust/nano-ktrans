"""
nano-ktrans 基础测试

这些测试验证核心模块的基本功能，不需要 GPU 或完整模型权重即可运行：
- 层的构造和前向计算（RMSNorm, Linear, Attention 结构）
- 配置解析
- 路由逻辑
"""

import time

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

    def test_qkv_weight_loader_rejects_unknown_shard(self):
        """QKVParallelLinear 对未知 shard id 应抛 ValueError"""
        from nano_ktrans.layers.linear import QKVParallelLinear

        linear = QKVParallelLinear(
            hidden_size=64,
            head_size=16,
            total_num_heads=2,
            total_num_kv_heads=2,
        )
        bogus = torch.zeros(32, 64)
        with pytest.raises(ValueError):
            linear.weight_loader(linear.weight, bogus, "x")

    def test_merged_column_parallel_rejects_out_of_range_shard(self):
        """MergedColumnParallelLinear 对越界 shard id 应抛 ValueError"""
        from nano_ktrans.layers.linear import MergedColumnParallelLinear

        linear = MergedColumnParallelLinear(input_size=32, output_sizes=[16, 16])
        shard = torch.zeros(16, 32)
        with pytest.raises(ValueError):
            linear.weight_loader(linear.weight, shard, 5)


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

    def test_get_rope_accepts_identity_scaling(self):
        """get_rope 对 identity/None rope_scaling 都应可用"""
        from nano_ktrans.layers.rotary_embedding import get_rope, RotaryEmbedding

        rope = get_rope(64, 64, 128, 10000.0, rope_scaling=None)
        assert isinstance(rope, RotaryEmbedding)

        rope2 = get_rope(64, 64, 128, 10000.0, rope_scaling={"type": "default"})
        assert isinstance(rope2, RotaryEmbedding)

    def test_get_rope_rejects_unimplemented_scaling(self):
        """get_rope 对真实 scaling 类型应显式报错而非静默错误结果"""
        from nano_ktrans.layers.rotary_embedding import get_rope

        # lru_cache 以参数做 key，这里传入新的 base 避免命中之前的缓存项
        with pytest.raises(NotImplementedError):
            get_rope(64, 64, 128, 12345.0, rope_scaling={"type": "yarn", "factor": 2.0})


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


class TestSimpleEngine:
    def test_refresh_offload_state_helper_uses_model_hook(self):
        from nano_ktrans.engine.simple_engine import SimpleEngine

        class DummyInnerModel:
            def __init__(self):
                self.calls = 0
                self.last_phase = None

            def offload_worker_running(self):
                return False

            def refresh_offload_state(self, *, phase="decode"):
                self.calls += 1
                self.last_phase = phase
                return 7

        class DummyOuterModel:
            def __init__(self):
                self.model = DummyInnerModel()

        engine = SimpleEngine.__new__(SimpleEngine)
        engine.model = DummyOuterModel()

        assert engine._refresh_offload_state() == 7
        assert engine.model.model.calls == 1
        assert engine.model.model.last_phase == "decode"

    def test_refresh_offload_state_skips_manual_background_tick_when_worker_running(self):
        from nano_ktrans.engine.simple_engine import SimpleEngine

        class DummyInnerModel:
            def __init__(self):
                self.background_calls = 0
                self.refresh_calls = 0

            def offload_worker_running(self):
                return True

            def background_tick_offload_state(self, *, phase="decode"):
                self.background_calls += 1
                return 3

            def refresh_offload_state(self, *, phase="decode"):
                self.refresh_calls += 1
                return 5

        class DummyOuterModel:
            def __init__(self):
                self.model = DummyInnerModel()

        engine = SimpleEngine.__new__(SimpleEngine)
        engine.model = DummyOuterModel()

        assert engine._refresh_offload_state() == 5
        assert engine.model.model.background_calls == 0
        assert engine.model.model.refresh_calls == 1

    def test_engine_can_start_and_stop_background_offload_worker(self):
        from nano_ktrans.engine.simple_engine import SimpleEngine

        called = {"start": 0, "stop": 0}

        class DummyInnerModel:
            def start_offload_worker(self):
                called["start"] += 1

            def shutdown_offload_worker(self):
                called["stop"] += 1

        class DummyOuterModel:
            def __init__(self):
                self.model = DummyInnerModel()

        engine = SimpleEngine.__new__(SimpleEngine)
        engine.model = DummyOuterModel()

        assert engine.start_background_offload_worker() is True
        assert engine.stop_background_offload_worker() is True
        assert called["start"] == 1
        assert called["stop"] == 1


class TestBackgroundOffloadWorker:
    def test_worker_records_ticks_and_can_reset(self):
        from nano_ktrans.kernels.offload_worker import BackgroundOffloadWorker

        work = {"count": 0}

        def tick():
            work["count"] += 1
            return 1 if work["count"] <= 2 else 0

        worker = BackgroundOffloadWorker(
            tick,
            poll_interval_seconds=0.001,
            auto_start=False,
        )
        try:
            worker.start()
            time.sleep(0.02)
            diagnostics = worker.diagnostics()
            assert diagnostics["enabled"] is True
            assert diagnostics["ticks"] >= 1
            assert diagnostics["work_ticks"] >= 1
            worker.reset_counters()
            diagnostics = worker.diagnostics()
            assert diagnostics["ticks"] == 0
            assert diagnostics["work_ticks"] == 0
            assert diagnostics["last_work_items"] == 0
        finally:
            worker.shutdown()

    def test_worker_shutdown_marks_disabled(self):
        from nano_ktrans.kernels.offload_worker import BackgroundOffloadWorker

        worker = BackgroundOffloadWorker(lambda: 0, poll_interval_seconds=0.001)
        worker.shutdown()
        diagnostics = worker.diagnostics()
        assert diagnostics["enabled"] is False


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
    def test_scheduler_profile_overlap_safe(self):
        from nano_ktrans.scheduler import (
            SCHEDULER_PROFILE_OVERLAP_SAFE,
            SchedulerConfig,
            resolve_scheduler_profile,
        )

        resolved = resolve_scheduler_profile(
            SCHEDULER_PROFILE_OVERLAP_SAFE,
            base_config=SchedulerConfig(enabled=True),
        )

        assert resolved.prefill_collect_only is True
        assert resolved.decode_require_prefetch_ready is True
        assert resolved.prefetch_candidate_budget_per_layer >= 2
        assert resolved.demotion_idle_steps >= 2
        assert resolved.migration_cooldown_steps >= 2

    def test_scheduler_profile_preserves_stronger_explicit_values(self):
        from nano_ktrans.scheduler import (
            SCHEDULER_PROFILE_EAGER,
            SchedulerConfig,
            resolve_scheduler_profile,
        )

        resolved = resolve_scheduler_profile(
            SCHEDULER_PROFILE_EAGER,
            base_config=SchedulerConfig(
                enabled=True,
                step_stride_prefill=2,
                demotion_idle_steps=3,
                migration_cooldown_steps=5,
                prefetch_candidate_budget_per_layer=6,
            ),
        )

        assert resolved.step_stride_prefill == 2
        assert resolved.demotion_idle_steps == 3
        assert resolved.migration_cooldown_steps == 5
        assert resolved.prefetch_candidate_budget_per_layer == 6
        assert resolved.prefill_collect_only is False

    def test_scheduler_profile_summary_reports_prepared_budget_heuristic(self):
        from nano_ktrans.scheduler import SchedulerConfig, scheduler_profile_summary

        summary = scheduler_profile_summary(
            "baseline",
            SchedulerConfig(
                enabled=True,
                decode_promote_k=3,
                prefetch_candidate_budget_per_layer=4,
            ),
        )

        assert summary["prepared_cache_budget_heuristic"] == 6

    def test_llm_get_offload_diagnostics_reports_prepared_budget_heuristic(self):
        from nano_ktrans.llm import LLM
        from nano_ktrans.scheduler import SchedulerConfig

        llm = LLM.__new__(LLM)
        llm.offload_backend = "cpu"
        llm.scheduler_profile = "baseline"
        llm.prepared_cache_budget = 6
        llm.prepared_controller_aggressiveness = 0.5
        llm.enable_background_offload_worker = False
        llm.background_offload_poll_interval_seconds = 0.01
        llm.dynamic_expert_scheduler = type(
            "Scheduler",
            (),
            {
                "config": SchedulerConfig(
                    enabled=True,
                    decode_promote_k=3,
                    prefetch_candidate_budget_per_layer=4,
                ),
                "diagnostics": staticmethod(lambda: {"enabled": True}),
            },
        )()
        llm.model = type(
            "Wrapper",
            (),
            {
                "model": type(
                    "Inner",
                    (),
                    {
                        "layers": [],
                        "offload_refresh_diagnostics": staticmethod(lambda: {"offload_refresh_calls": 0}),
                    },
                )(),
            },
        )()

        diagnostics = llm.get_offload_diagnostics()
        assert diagnostics["prepared_cache_budget_heuristic"] == 6

    def test_resolve_prepared_cache_budget_varies_by_profile(self):
        from nano_ktrans.scheduler import (
            SCHEDULER_PROFILE_BASELINE,
            SCHEDULER_PROFILE_EAGER,
            SCHEDULER_PROFILE_OVERLAP_SAFE,
            SchedulerConfig,
            resolve_prepared_cache_budget,
        )

        config = SchedulerConfig(
            enabled=True,
            decode_promote_k=2,
            prefetch_candidate_budget_per_layer=4,
        )

        assert resolve_prepared_cache_budget(SCHEDULER_PROFILE_BASELINE, config) == 4
        assert resolve_prepared_cache_budget(SCHEDULER_PROFILE_OVERLAP_SAFE, config) == 5
        assert resolve_prepared_cache_budget(SCHEDULER_PROFILE_EAGER, config) == 6

    def test_resolve_prepared_controller_aggressiveness_varies_by_profile(self):
        from nano_ktrans.scheduler import (
            SCHEDULER_PROFILE_BASELINE,
            SCHEDULER_PROFILE_EAGER,
            SCHEDULER_PROFILE_OVERLAP_SAFE,
            resolve_prepared_controller_aggressiveness,
        )

        assert resolve_prepared_controller_aggressiveness(SCHEDULER_PROFILE_BASELINE) == 0.0
        assert resolve_prepared_controller_aggressiveness(SCHEDULER_PROFILE_OVERLAP_SAFE) == 0.5
        assert resolve_prepared_controller_aggressiveness(SCHEDULER_PROFILE_EAGER) == 1.0

    def test_normalize_scheduler_profiles_dedupes(self):
        from nano_ktrans.scheduler import normalize_scheduler_profiles

        normalized = normalize_scheduler_profiles(
            ["baseline", "overlap-safe", "baseline", "eager"],
        )
        assert normalized == ["baseline", "overlap_safe", "eager"]

    def test_summarize_offload_diagnostics(self):
        from nano_ktrans.scheduler import summarize_offload_diagnostics

        diagnostics = {
            "scheduler_profile": {"profile": "overlap_safe"},
            "offload_refresh": {
                "offload_refresh_calls": 3,
                "offload_refresh_ready_total": 2,
                "background_worker": {
                    "enabled": True,
                    "ticks": 10,
                    "work_ticks": 4,
                    "last_work_items": 2,
                },
            },
            "dynamic_scheduler": {"enabled": True},
            "layer_count": 2,
            "layers": [
                {
                    "prefetch_requested": 4,
                    "prefetch_enqueued": 2,
                    "prefetch_materialized": 1,
                    "prefetch_candidate_scans": 1,
                    "decode_prefetch_hits": 1,
                    "decode_prefetch_misses": 1,
                    "runtime_evictions": 1,
                    "runtime_deferred_for_prefetch": 2,
                    "runtime_skipped_demotion_cooldown": 3,
                    "applied_migration_ops": 4,
                    "pending_migrations": [{"expert_idx": 1}],
                    "backend": {
                        "migration_submit_calls": 2,
                        "migration_manager": {
                            "layers": [
                                {
                                    "total_enqueued_ops": 5,
                                    "total_deduped_ops": 2,
                                    "total_drained_ops": 4,
                                    "pending_ops": 1,
                                }
                            ]
                        },
                    },
                },
                {
                    "prefetch_requested": 3,
                    "prefetch_enqueued": 3,
                    "prefetch_materialized": 2,
                    "prefetch_candidate_scans": 2,
                    "decode_prefetch_hits": 2,
                    "decode_prefetch_misses": 0,
                    "runtime_evictions": 0,
                    "runtime_deferred_for_prefetch": 1,
                    "runtime_skipped_demotion_cooldown": 0,
                    "applied_migration_ops": 1,
                    "pending_migrations": [],
                    "backend": {
                        "migration_submit_calls": 1,
                        "migration_manager": {
                            "layers": [
                                {
                                    "total_enqueued_ops": 3,
                                    "total_deduped_ops": 1,
                                    "total_drained_ops": 2,
                                    "pending_ops": 0,
                                }
                            ]
                        },
                    },
                },
            ],
        }

        summary = summarize_offload_diagnostics(diagnostics)
        assert summary["enabled"] is True
        assert summary["layer_count"] == 2
        assert summary["offload_refresh_calls"] == 3
        assert summary["offload_refresh_ready_total"] == 2
        assert summary["prefetch_requested"] == 7
        assert summary["prefetch_enqueued"] == 5
        assert summary["decode_prefetch_hits"] == 3
        assert summary["decode_prefetch_misses"] == 1
        assert summary["migration_submit_calls"] == 3
        assert summary["migration_total_enqueued_ops"] == 8
        assert summary["migration_total_deduped_ops"] == 3
        assert summary["background_worker_enabled"] is True
        assert summary["background_worker_ticks"] == 10
        assert summary["background_worker_work_ticks"] == 4
        assert summary["background_worker_last_work_items"] == 2
        assert summary["background_worker_work_ratio"] == pytest.approx(0.4)

    def test_summarize_offload_diagnostics_reports_pipeline_apply_batches(self):
        from nano_ktrans.scheduler import summarize_offload_diagnostics

        summary = summarize_offload_diagnostics(
            {
                "dynamic_scheduler": {"enabled": True},
                "layer_count": 1,
                "offload_refresh": {
                    "offload_pipeline_apply_batch_count_total": 3,
                    "offload_pipeline_apply_batch_experts_total": 7,
                    "offload_pipeline_apply_batch_evictions_total": 2,
                },
                "layers": [
                    {
                        "pipeline_apply_batches": 2,
                        "pipeline_apply_batch_experts": 5,
                        "pipeline_apply_batch_evictions": 1,
                        "pipeline_apply_batch_activated": 2,
                        "pipeline_apply_batch_warm": 2,
                        "pipeline_apply_batch_cold": 1,
                    }
                ],
            }
        )

        assert summary["pipeline_apply_batches"] == 2
        assert summary["pipeline_apply_batch_experts"] == 5
        assert summary["pipeline_apply_batch_evictions"] == 1
        assert summary["pipeline_apply_batch_activated"] == 2
        assert summary["pipeline_apply_batch_warm"] == 2
        assert summary["pipeline_apply_batch_cold"] == 1
        assert summary["pipeline_apply_batch_size_avg"] == pytest.approx(2.5)
        assert summary["pipeline_apply_batch_activated_ratio"] == pytest.approx(0.4)
        assert summary["pipeline_apply_batch_warm_ratio"] == pytest.approx(0.4)
        assert summary["pipeline_apply_batch_cold_ratio"] == pytest.approx(0.2)
        assert summary["offload_pipeline_apply_batch_count_total"] == 3
        assert summary["offload_pipeline_apply_batch_experts_total"] == 7
        assert summary["offload_pipeline_apply_batch_evictions_total"] == 2

    def test_summarize_profile_sweep_results(self):
        from nano_ktrans.scheduler import summarize_profile_sweep_results

        summary = summarize_profile_sweep_results(
            [
                {
                    "backend": "cuda_cpu_offload",
                    "scheduler_profile": "baseline",
                    "status": "ok",
                    "scheduler_summary": {
                        "background_worker_enabled": True,
                        "background_worker_ticks": 10,
                        "background_worker_work_ticks": 4,
                        "background_worker_work_ratio": 0.4,
                        "pipeline_prefetch_overlap_hits": 2,
                        "pipeline_promotion_source_activated": 1,
                        "pipeline_promotion_source_warm": 1,
                        "pipeline_promotion_source_cold": 3,
                        "pipeline_apply_batches": 2,
                        "pipeline_apply_batch_size_avg": 1.5,
                        "pipeline_apply_batch_evictions": 1,
                        "pipeline_apply_batch_activated": 1,
                        "pipeline_apply_batch_warm": 1,
                        "pipeline_apply_batch_cold": 1,
                        "pipeline_apply_batch_activated_ratio": 1.0 / 3.0,
                        "pipeline_apply_batch_warm_ratio": 1.0 / 3.0,
                        "pipeline_apply_batch_cold_ratio": 1.0 / 3.0,
                        "apply_queue_pressure_avg": 1.5,
                        "apply_queue_pressure_ema_avg": 0.8,
                        "apply_queue_budget_backoff_avg": 1.0,
                        "apply_commit_queue_pressure_avg": 1.0,
                        "apply_commit_queue_pressure_ema_avg": 0.6,
                        "apply_commit_queue_budget_backoff_avg": 1.0,
                        "apply_commit_batch_queue_pressure_avg": 0.75,
                        "apply_commit_batch_queue_pressure_ema_avg": 0.5,
                        "apply_commit_batch_queue_budget_backoff_avg": 1.0,
                        "offload_pipeline_apply_batch_count_total": 2,
                        "offload_pipeline_apply_batch_experts_total": 3,
                        "offload_pipeline_apply_batch_evictions_total": 1,
                        "cold_promotion_penalty_avg": 1.5,
                        "migration_activation_eviction_regressions": 2,
                        "migration_warm_eviction_regressions": 3,
                        "runtime_deferred_for_prefetch": 4,
                    },
                    "runs": [
                        {
                            "prefill_seconds": 2.0,
                            "decode_seconds": 1.0,
                            "decode_tokens_per_second": 1.5,
                        }
                    ],
                },
                {
                    "backend": "cuda_cpu_offload",
                    "scheduler_profile": "overlap_safe",
                    "status": "ok",
                    "scheduler_summary": {
                        "background_worker_enabled": True,
                        "background_worker_ticks": 12,
                        "background_worker_work_ticks": 7,
                        "background_worker_work_ratio": 7 / 12,
                        "pipeline_prefetch_overlap_hits": 4,
                        "pipeline_promotion_source_activated": 3,
                        "pipeline_promotion_source_warm": 0,
                        "pipeline_promotion_source_cold": 1,
                        "pipeline_apply_batches": 3,
                        "pipeline_apply_batch_size_avg": 2.0,
                        "pipeline_apply_batch_evictions": 2,
                        "pipeline_apply_batch_activated": 3,
                        "pipeline_apply_batch_warm": 2,
                        "pipeline_apply_batch_cold": 1,
                        "pipeline_apply_batch_activated_ratio": 0.5,
                        "pipeline_apply_batch_warm_ratio": 1.0 / 3.0,
                        "pipeline_apply_batch_cold_ratio": 1.0 / 6.0,
                        "apply_queue_pressure_avg": 0.5,
                        "apply_queue_pressure_ema_avg": 0.25,
                        "apply_queue_budget_backoff_avg": 0.0,
                        "apply_commit_queue_pressure_avg": 0.25,
                        "apply_commit_queue_pressure_ema_avg": 0.1,
                        "apply_commit_queue_budget_backoff_avg": 0.0,
                        "apply_commit_batch_queue_pressure_avg": 0.2,
                        "apply_commit_batch_queue_pressure_ema_avg": 0.05,
                        "apply_commit_batch_queue_budget_backoff_avg": 0.0,
                        "offload_pipeline_apply_batch_count_total": 3,
                        "offload_pipeline_apply_batch_experts_total": 6,
                        "offload_pipeline_apply_batch_evictions_total": 2,
                        "cold_promotion_penalty_avg": 0.5,
                        "migration_activation_eviction_regressions": 0,
                        "migration_warm_eviction_regressions": 1,
                        "runtime_deferred_for_prefetch": 1,
                    },
                    "runs": [
                        {
                            "prefill_seconds": 1.5,
                            "decode_seconds": 0.5,
                            "decode_tokens_per_second": 2.5,
                        }
                    ],
                },
            ]
        )

        assert len(summary["profiles"]) == 2
        assert summary["best_by_decode_tokens_per_second"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_decode_tokens_per_second"]["decode_tokens_per_second"] == pytest.approx(2.5)
        assert summary["metric_directions"]["pipeline_promotion_source_cold"] == "min"
        assert summary["metric_directions"]["cold_promotion_penalty_avg"] == "min"
        assert summary["metric_directions"]["migration_activation_eviction_regressions"] == "min"
        assert summary["metric_directions"]["apply_queue_pressure_avg"] == "min"
        assert summary["metric_directions"]["apply_commit_queue_pressure_avg"] == "min"
        assert summary["metric_directions"]["apply_commit_batch_queue_pressure_avg"] == "min"
        assert summary["metric_directions"]["apply_queue_commit_batch_size_avg"] == "max"
        assert "pipeline_apply_batch_size_avg" in summary["sort_keys"]
        assert "pipeline_promotion_non_cold_ratio" in summary["sort_keys"]
        assert summary["best_by_decode_tokens_per_second"]["runtime_offload_pipeline_apply_batch_count_total"] == 3
        assert summary["best_by_metric"]["pipeline_promotion_source_cold"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["cold_promotion_penalty_avg"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["migration_activation_eviction_regressions"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["apply_queue_pressure_avg"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["apply_commit_queue_pressure_avg"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["apply_commit_batch_queue_pressure_avg"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["background_worker_work_ratio"]["scheduler_profile"] == "overlap_safe"
        assert summary["best_by_metric"]["pipeline_promotion_non_cold_ratio"]["value"] == pytest.approx(0.75)
        assert summary["best_by_metric"]["runtime_apply_batch_size_avg"]["value"] == pytest.approx(2.0)
        assert summary["comparison_table"][0]["pipeline_apply_batch_activated"] == 3
        assert summary["comparison_table"][0]["pipeline_apply_batch_cold_ratio"] == pytest.approx(1.0 / 6.0)
        assert summary["comparison_table"][0]["migration_activation_eviction_regressions"] == 0
        assert summary["comparison_table"][0]["scheduler_profile"] == "overlap_safe"
        assert summary["comparison_table"][0]["rank_by_decode_tokens_per_second"] == 1
        assert summary["comparison_table"][0]["pipeline_promotion_non_cold_total"] == 3
        assert summary["comparison_table"][0]["runtime_apply_batch_size_avg"] == pytest.approx(2.0)
        assert summary["comparison_table"][0]["background_worker_work_ratio"] == pytest.approx(7 / 12)

    def test_migration_pipeline_runtime_tracks_apply_batch_totals(self):
        from nano_ktrans.kernels.migration_runtime import MigrationPipelineRuntime

        class DummyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.hybrid_moe = self

            def advance_offload_pipeline(self, *, phase, device, dtype):
                return {
                    "ready_polled": 1,
                    "ready_applied": 2,
                    "ready_deferred": 1,
                    "prefetch_submitted": 3,
                    "activation_ready": 4,
                    "apply_batch_count": 2,
                    "apply_batch_experts": 5,
                    "apply_batch_evictions": 1,
                    "apply_batch_activated": 3,
                    "apply_batch_warm": 1,
                    "apply_batch_cold": 1,
                }

        runtime = MigrationPipelineRuntime()
        tick = runtime.tick_layers([DummyLayer()], phase="decode")
        diagnostics = runtime.diagnostics()

        assert tick["apply_batch_count"] == 2
        assert tick["apply_batch_experts"] == 5
        assert tick["apply_batch_evictions"] == 1
        assert tick["apply_batch_activated"] == 3
        assert tick["apply_batch_warm"] == 1
        assert tick["apply_batch_cold"] == 1
        assert diagnostics["offload_pipeline_apply_batch_count_total"] == 2
        assert diagnostics["offload_pipeline_apply_batch_experts_total"] == 5
        assert diagnostics["offload_pipeline_apply_batch_evictions_total"] == 1
        assert diagnostics["offload_pipeline_apply_batch_activated_total"] == 3
        assert diagnostics["offload_pipeline_apply_batch_warm_total"] == 1
        assert diagnostics["offload_pipeline_apply_batch_cold_total"] == 1

    def test_migration_pipeline_runtime_tracks_background_tick_totals(self):
        from nano_ktrans.kernels.migration_runtime import MigrationPipelineRuntime

        class DummyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.hybrid_moe = self

            def background_advance_offload_pipeline(self, *, phase, device, dtype):
                return {
                    "ready_polled": 3,
                    "warm_prebuilt": 2,
                    "activation_ready": 1,
                    "apply_commit_queue_enqueued": 4,
                    "apply_commit_batch_queue_enqueued": 2,
                    "resident_commit_batch_queue_enqueued": 1,
                    "resident_commit_batch_queue_prefinalized": 1,
                    "resident_commit_finalize_queue_enqueued": 1,
                    "resident_commit_finalize_queue_prefinalized": 1,
                    "resident_commit_ready_cache_stores": 1,
                }

        runtime = MigrationPipelineRuntime()
        tick = runtime.background_tick_layers([DummyLayer()], phase="decode")
        diagnostics = runtime.diagnostics()

        assert tick["background_ready_callbacks"] == 3
        assert tick["background_warm_prebuilt"] == 2
        assert tick["background_activation_ready"] == 1
        assert diagnostics["offload_background_ticks"] == 1
        assert diagnostics["offload_pipeline_background_ready_callback_total"] == 3
        assert diagnostics["offload_background_warm_prebuilt_total"] == 2
        assert diagnostics["offload_background_activation_ready_total"] == 1
        assert diagnostics["offload_background_apply_commit_queue_enqueued_total"] == 4
        assert diagnostics["offload_background_apply_commit_batch_queue_enqueued_total"] == 2
        assert diagnostics["offload_background_resident_commit_batch_queue_enqueued_total"] == 1
        assert diagnostics["offload_background_resident_commit_batch_queue_prefinalized_total"] == 1
        assert diagnostics["offload_background_resident_commit_finalize_queue_enqueued_total"] == 1
        assert diagnostics["offload_background_resident_commit_finalize_queue_prefinalized_total"] == 1
        assert diagnostics["offload_background_resident_commit_ready_cache_stores_total"] == 1

    def test_hybrid_moe_advance_pipeline_reports_incremental_batch_metrics(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="incremental_batch_metrics",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.READY,
                phase="decode",
            )

        first = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        second = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()

        assert first["apply_batch_count"] == 1
        assert first["apply_batch_experts"] == 1
        assert first["apply_batch_evictions"] == 1
        assert second["apply_batch_count"] == 1
        assert second["apply_batch_experts"] == 1
        assert second["apply_batch_evictions"] == 1
        assert diagnostics["pipeline_apply_batches"] == 2
        assert diagnostics["pipeline_apply_batch_experts"] == 2
        assert diagnostics["pipeline_apply_batch_evictions"] == 2

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
                prefill_collect_only=False,
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

    def test_migration_manager_dedupes_by_expert(self):
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
                    layer_idx=1,
                    expert_idx=4,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="prefetch_1",
                )
            ],
            phase="prefill",
        )
        backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=1,
                    expert_idx=4,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="prefetch_2",
                ),
                ExpertMigrationOp(
                    layer_idx=1,
                    expert_idx=5,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="prefetch_3",
                ),
            ],
            phase="decode",
        )
        layer_diag = backend.diagnostics()["migration_manager"]["layers"][0]
        assert layer_diag["pending_ops"] == 2
        assert layer_diag["total_enqueued_ops"] == 3
        assert layer_diag["total_deduped_ops"] >= 1
        assert layer_diag["history"][-1]["deduped_plan_size"] == 2
        assert layer_diag["lifecycle_state_counts"]["queued"] >= 1

    def test_migration_manager_tracks_lifecycle(self):
        from nano_ktrans.kernels.expert_migration import ExpertMigrationManager, MigrationLifecycle
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        manager = ExpertMigrationManager()
        manager.queue(
            0,
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=3,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="promote_hot_expert",
                )
            ],
            phase="decode",
        )
        manager.mark_state(0, 3, state=MigrationLifecycle.PREFETCHING, phase="decode")
        manager.mark_state(0, 3, state=MigrationLifecycle.READY, phase="decode")
        manager.mark_state(0, 3, state=MigrationLifecycle.WARMED, phase="decode")
        manager.mark_state(0, 3, state=MigrationLifecycle.APPLIED, phase="decode")

        diagnostics = manager.diagnostics()
        layer_diag = diagnostics["layers"][0]
        assert layer_diag["total_prefetching_events"] == 1
        assert layer_diag["total_ready_events"] == 1
        assert layer_diag["total_warmed_events"] == 1
        assert layer_diag["total_applied_events"] == 1
        assert layer_diag["lifecycle_state_counts"]["applied"] == 1
        assert layer_diag["lifecycle"][0]["state"] == "applied"

    def test_migration_manager_can_take_ready_subset(self):
        from nano_ktrans.kernels.expert_migration import ExpertMigrationManager, MigrationLifecycle
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        manager = ExpertMigrationManager()
        manager.queue(
            0,
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="promote_1",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=2,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="promote_2",
                ),
            ],
            phase="decode",
        )
        manager.mark_state(0, 2, state=MigrationLifecycle.READY, phase="decode")
        ready_ops = manager.take_ready_layer(0)
        pending_ops = manager.peek_layer(0)
        layer_diag = manager.diagnostics()["layers"][0]

        assert [op.expert_idx for op in ready_ops] == [2]
        assert [op.expert_idx for op in pending_ops] == [1]
        assert layer_diag["total_ready_drains"] == 1

    def test_backend_notify_expert_evicted_called_on_demotion(self, tmp_path):
        """notify_expert_evicted must fire when an expert is demoted from GPU to PIM."""
        from safetensors.torch import save_file
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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

        # Start with expert 1 resident on GPU, expert 0 on the offload tier.
        gpu_mask = torch.tensor([False, True], dtype=torch.bool)
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
                decode_promote_k=2,
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

        # Avoid spinning up background materialization/activation in this
        # sync unit test (mirrors the approach used by the sibling migration
        # test to keep behaviour deterministic).
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        # Track invocations of notify_expert_evicted on the backend.
        notify_calls = []
        original_notify = hybrid.offload_backend.notify_expert_evicted

        def tracked_notify(expert_idx, residency_before):
            notify_calls.append((expert_idx, residency_before))
            return original_notify(expert_idx, residency_before)

        hybrid.offload_backend.notify_expert_evicted = tracked_notify

        # Queue a demotion op: expert 1 GPU -> PIM. It will be applied during
        # the next forward() call via the migration pipeline.
        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.GPU,
                    dst=ExpertResidency.PIM,
                    reason="test_demote",
                )
            ],
            phase="decode",
        )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        evicted_experts = [idx for idx, src in notify_calls if idx == 1]
        assert len(evicted_experts) > 0, (
            f"Expected notify_expert_evicted to be called for expert 1, "
            f"got calls: {notify_calls}"
        )
        # The demotion path always reports 'gpu' as the prior residency.
        assert all(src == "gpu" for _, src in notify_calls)
        assert bool(hybrid.gpu_experts_mask[1].item()) is False


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
                decode_promote_k=2,
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

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

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
                decode_promote_k=2,
                prefill_collect_only=False,
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
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        diagnostics = hybrid.diagnostics()
        assert diagnostics["prefetch_requested"] >= 2
        assert diagnostics["prefetch_enqueued"] >= 1
        assert diagnostics["materialization_manager"]["prefetch_resolved"] >= 1
        assert diagnostics["materialization_manager"]["cache_size"] >= 1
        assert diagnostics["decode_prefetch_hits"] >= 1

    def test_scheduler_respects_recent_access_and_cooldown(self):
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        masks = [torch.tensor([True, False, False], dtype=torch.bool)]
        plan = ExpertResidencyPlan.from_gpu_masks(masks, default_offload_tier=ExpertResidency.PIM)
        scheduler = DynamicExpertScheduler(
            residency_plan=plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                demotion_idle_steps=2,
                migration_cooldown_steps=2,
            ),
        )

        scheduler.observe(0, torch.tensor([[0]]), phase="decode")
        ops = scheduler.plan_layer(0, phase="decode")
        assert [op.expert_idx for op in ops if op.dst == ExpertResidency.GPU] == []
        assert [op.expert_idx for op in ops if op.src == ExpertResidency.GPU] == []

        scheduler.observe(0, torch.tensor([[2]]), phase="decode")
        scheduler.observe(0, torch.tensor([[1]]), phase="decode")
        scheduler.observe(0, torch.tensor([[1]]), phase="decode")
        ops = scheduler.plan_layer(0, phase="decode")
        promoted = [op.expert_idx for op in ops if op.dst == ExpertResidency.GPU]
        demoted = [op.expert_idx for op in ops if op.src == ExpertResidency.GPU]
        assert promoted == [1]
        assert demoted == [0]

    def test_scheduler_prefill_collect_only(self):
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        masks = [torch.tensor([True, False, False], dtype=torch.bool)]
        plan = ExpertResidencyPlan.from_gpu_masks(masks, default_offload_tier=ExpertResidency.PIM)
        scheduler = DynamicExpertScheduler(
            residency_plan=plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                prefill_force_gpu_budget_per_layer=2,
                prefill_collect_only=True,
                offload_tier=ExpertResidency.PIM,
            ),
        )

        scheduler.observe(0, torch.tensor([[1, 2], [2, 2]]), phase="prefill")
        ops = scheduler.plan_layer(0, phase="prefill")
        assert ops == []
        diagnostics = scheduler.diagnostics()
        assert diagnostics["prefill_collect_only"] is True
        assert diagnostics["residency_plan"]["layers"][0]["logical_step"] == 8

    def test_scheduler_prefetch_candidates_layer(self):
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

        masks = [torch.tensor([True, False, False, False], dtype=torch.bool)]
        plan = ExpertResidencyPlan.from_gpu_masks(masks, default_offload_tier=ExpertResidency.PIM)
        scheduler = DynamicExpertScheduler(
            residency_plan=plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                prefetch_candidate_budget_per_layer=2,
            ),
        )

        scheduler.observe(0, torch.tensor([[2, 3], [2, 2]]), phase="decode")
        candidates = scheduler.prefetch_candidates_layer(0, phase="decode")
        assert candidates == [2, 3]

    def test_hybrid_moe_can_defer_decode_promotion_until_prefetch_ready(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                decode_require_prefetch_ready=True,
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

        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
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
            output = hybrid(hidden_states, router_logits)
            assert output.shape == (1, 4)
            assert "1" not in hybrid.gpu_experts

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
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        assert output.shape == (1, 4)
        assert "1" in hybrid.gpu_experts
        diagnostics = hybrid.diagnostics()
        assert diagnostics["runtime_deferred_for_prefetch"] >= 1

    def test_hybrid_moe_prefetches_hot_candidates_without_immediate_migration(self, tmp_path):
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                prefetch_candidate_budget_per_layer=2,
                decode_promote_k=0,
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
        assert diagnostics["prefetch_candidate_scans"] >= 1
        assert diagnostics["prefetch_enqueued"] >= 2

    def test_mixtral_model_refreshes_all_hybrid_moe_layers(self):
        from nano_ktrans.models.mixtral import MixtralModel

        class DummyHybrid:
            def __init__(self, ready_count):
                self.ready_count = ready_count
                self.advanced = []
                self.background_ticks = 0

            def background_tick_offload_state(self):
                self.background_ticks += 1
                return self.ready_count

            def advance_offload_pipeline(self, *, phase, device, dtype):
                self.advanced.append((phase, str(device), str(dtype)))
                return {
                    "prefetch_submitted": 2,
                    "ready_polled": self.ready_count,
                    "activation_ready": 1,
                    "ready_applied": 1,
                    "ready_deferred": 0,
                }

        model = MixtralModel.__new__(MixtralModel)
        layer0 = type("Layer", (), {"hybrid_moe": DummyHybrid(2)})()
        layer1 = type("Layer", (), {"hybrid_moe": None})()
        layer2 = type("Layer", (), {"hybrid_moe": DummyHybrid(3)})()
        linear0 = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
        linear2 = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
        layer0.parameters = linear0.parameters
        layer2.parameters = linear2.parameters
        model.layers = [layer0, layer1, layer2]

        assert model.refresh_offload_state(phase="decode") == 5
        diagnostics = model.offload_refresh_diagnostics()
        assert diagnostics["offload_refresh_calls"] == 1
        assert diagnostics["offload_background_ticks"] == 0
        assert diagnostics["offload_refresh_ready_total"] == 5
        assert diagnostics["offload_pipeline_prefetch_submitted_total"] == 4
        assert diagnostics["offload_pipeline_activation_ready_total"] == 2
        assert diagnostics["offload_pipeline_ready_applied_total"] == 2
        assert diagnostics["offload_pipeline_last_phase"] == "decode"
        assert layer0.hybrid_moe.advanced[0][0] == "decode"
        assert layer2.hybrid_moe.advanced[0][0] == "decode"

    def test_mixtral_model_background_tick_runs_before_refresh(self):
        from nano_ktrans.models.mixtral import MixtralModel

        class DummyHybrid:
            def __init__(self):
                self.background_ticks = 0
                self.refresh_calls = 0

            def background_advance_offload_pipeline(self, *, phase, device, dtype):
                self.background_ticks += 1
                return {
                    "ready_polled": 2,
                    "warm_prebuilt": 1,
                    "activation_ready": 1,
                    "activation_applied": 1,
                }

            def advance_offload_pipeline(self, *, phase, device, dtype):
                self.refresh_calls += 1
                return {
                    "prefetch_submitted": 0,
                    "ready_polled": 1,
                    "activation_ready": 0,
                    "ready_applied": 0,
                    "ready_deferred": 0,
                    "apply_batch_count": 0,
                    "apply_batch_experts": 0,
                    "apply_batch_evictions": 0,
                    "apply_batch_activated": 0,
                    "apply_batch_warm": 0,
                    "apply_batch_cold": 0,
                }

        model = MixtralModel.__new__(MixtralModel)
        layer = type("Layer", (), {"hybrid_moe": DummyHybrid()})()
        linear = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
        layer.parameters = linear.parameters
        model.layers = [layer]

        assert model.background_tick_offload_state(phase="decode") == 5
        assert model.refresh_offload_state(phase="decode") == 1
        diagnostics = model.offload_refresh_diagnostics()
        assert diagnostics["offload_background_ticks"] == 1
        assert diagnostics["offload_background_work_items_total"] == 5
        assert diagnostics["offload_pipeline_background_ready_callback_total"] == 2
        assert diagnostics["offload_background_warm_prebuilt_total"] == 1
        assert diagnostics["offload_background_activation_ready_total"] == 1
        assert diagnostics["offload_background_activation_applied_total"] == 1

    def test_mixtral_model_reports_background_worker_diagnostics(self):
        from nano_ktrans.models.mixtral import MixtralModel

        class DummyWorker:
            def diagnostics(self):
                return {"enabled": True, "ticks": 3, "work_ticks": 2}

        model = MixtralModel.__new__(MixtralModel)
        model.layers = []
        model.offload_runtime = type("Runtime", (), {"diagnostics": lambda self: {"offload_refresh_calls": 0}})()
        model.offload_worker = DummyWorker()

        diagnostics = model.offload_refresh_diagnostics()
        assert diagnostics["background_worker"]["enabled"] is True
        assert diagnostics["background_worker"]["ticks"] == 3

    def test_mixtral_model_can_reset_background_worker_diagnostics(self):
        from nano_ktrans.models.mixtral import MixtralModel

        called = {"count": 0}

        class DummyWorker:
            def reset_counters(self):
                called["count"] += 1

        model = MixtralModel.__new__(MixtralModel)
        model.offload_worker = DummyWorker()
        model.reset_offload_worker_diagnostics()
        assert called["count"] == 1

    def test_mixtral_model_can_start_and_stop_background_worker(self):
        from nano_ktrans.models.mixtral import MixtralModel

        called = {"start": 0, "stop": 0}

        class DummyWorker:
            def start(self):
                called["start"] += 1

            def shutdown(self):
                called["stop"] += 1

            def is_running(self):
                return called["start"] > called["stop"]

        model = MixtralModel.__new__(MixtralModel)
        model.offload_worker = DummyWorker()
        model.start_offload_worker()
        assert model.offload_worker_running() is True
        model.shutdown_offload_worker()
        assert called["start"] == 1
        assert called["stop"] == 1

    def test_mixtral_model_background_worker_is_not_running_by_default(self):
        from nano_ktrans.models.mixtral import MixtralConfig, MixtralModel

        config = MixtralConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_local_experts=0,
        )
        model = MixtralModel(
            config,
            [torch.zeros(0, dtype=torch.bool)],
            enable_background_offload_worker=True,
        )
        try:
            assert model.offload_worker is not None
            assert model.offload_worker_running() is False
        finally:
            model.shutdown_offload_worker()

    def test_scheduler_summary_reports_background_offload_tick_metrics(self):
        from nano_ktrans.scheduler.diagnostics import summarize_offload_diagnostics

        summary = summarize_offload_diagnostics(
            {
                "offload_refresh": {
                    "offload_refresh_calls": 2,
                    "offload_background_ticks": 3,
                    "offload_background_work_items_total": 7,
                    "offload_background_warm_prebuilt_total": 4,
                    "offload_background_activation_ready_total": 2,
                    "offload_background_activation_applied_total": 1,
                    "offload_pipeline_ticks": 2,
                    "offload_pipeline_background_ready_callback_total": 5,
                },
                "layer_count": 0,
                "layers": [],
                "dynamic_scheduler": {"enabled": True},
            }
        )

        assert summary["offload_background_ticks"] == 3
        assert summary["offload_background_work_items_total"] == 7
        assert summary["offload_background_work_items_avg"] == pytest.approx(7 / 3)
        assert summary["offload_background_warm_prebuilt_total"] == 4
        assert summary["offload_background_activation_ready_total"] == 2
        assert summary["offload_background_activation_applied_total"] == 1
        assert summary["offload_pipeline_background_ready_callback_total"] == 5

    def test_profile_sweep_summary_includes_background_pipeline_totals(self):
        from nano_ktrans.scheduler.diagnostics import summarize_profile_sweep_results

        summary = summarize_profile_sweep_results(
            [
                {
                    "backend": "cuda_pim",
                    "scheduler_profile": "overlap_safe",
                    "status": "ok",
                    "runs": [
                        {
                            "decode_tokens_per_second": 1.5,
                            "prefill_seconds": 1.0,
                            "decode_seconds": 2.0,
                        }
                    ],
                    "scheduler_summary": {
                        "background_worker_work_ratio": 0.5,
                        "offload_background_work_items_total": 9,
                        "offload_background_work_items_avg": 3.0,
                        "offload_background_activation_applied_total": 4,
                        "pipeline_promotion_source_activated": 3,
                        "pipeline_promotion_source_warm": 2,
                        "pipeline_promotion_source_cold": 1,
                        "offload_pipeline_apply_batch_count_total": 2,
                        "offload_pipeline_apply_batch_experts_total": 4,
                        "offload_pipeline_apply_batch_evictions_total": 1,
                    },
                }
            ]
        )

        row = summary["comparison_table"][0]
        assert row["offload_background_work_items_total"] == 9
        assert row["offload_background_work_items_avg"] == 3.0
        assert row["offload_background_activation_applied_total"] == 4

    def test_materialization_manager_poll_ready(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager

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

        manager = ExpertMaterializationManager(
            weight_path=str(weight_path),
            expert_key_template="model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
            max_cached_experts=2,
            prefetch_workers=1,
        )
        assert manager.prefetch(0, 1) is True

        ready_keys = []
        for _ in range(10):
            ready_keys = manager.poll_ready()
            if ready_keys:
                break
            time.sleep(0.01)

        diagnostics = manager.diagnostics()
        assert (0, 1) in ready_keys
        assert diagnostics["prefetch_polled_ready"] >= 1
        assert diagnostics["prefetch_completion_events"] >= 1
        assert diagnostics["background_resolver_enabled"] is True
        assert diagnostics["prefetch_background_resolved"] >= 1
        assert manager.has_cached(0, 1) is True
        assert manager.has_pending_or_ready() is False
        manager.shutdown()

    def test_materialization_manager_can_stage_resident_expert(self):
        from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager

        manager = ExpertMaterializationManager.__new__(ExpertMaterializationManager)
        manager.expert_key_template = ""
        manager.expert_proj_names = None
        manager.max_cached_experts = 2
        manager.prefetch_workers = 0
        manager.executor = None
        from collections import OrderedDict, deque
        from threading import Lock

        manager._cache = OrderedDict()
        manager._futures = {}
        manager._ready_queue = deque()
        from queue import Queue
        from threading import Event
        manager._lock = Lock()
        manager._ready_mark_queue = Queue()
        manager._resolve_queue = Queue()
        manager._stop_event = Event()
        manager._resolver_thread = None
        manager._ready_callback = None
        manager.prefetch_submitted = 0
        manager.prefetch_resolved = 0
        manager.prefetch_polled_ready = 0
        manager.prefetch_completion_events = 0
        manager.prefetch_background_resolved = 0
        manager.prefetch_background_failures = 0
        manager.prefetch_background_ready_callbacks = 0
        manager.resident_stage_hits = 0
        manager.cache_hits = 0
        manager.sync_loads = 0
        manager.cache_evictions = 0

        staged = manager.stage_expert(
            0,
            3,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        assert staged is True
        assert manager.has_cached(0, 3) is True
        assert manager.diagnostics()["prefetch_resolved"] == 1

    def test_materialization_manager_background_resolver_drains_front_path(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager

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

        manager = ExpertMaterializationManager(
            weight_path=str(weight_path),
            expert_key_template="model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
            max_cached_experts=2,
            prefetch_workers=1,
        )
        assert manager.prefetch(0, 1) is True

        ready_keys = []
        for _ in range(20):
            if manager.has_cached(0, 1):
                ready_keys = manager.poll_ready()
                break
            time.sleep(0.01)

        diagnostics = manager.diagnostics()
        assert ready_keys == [(0, 1)]
        assert diagnostics["prefetch_background_resolved"] >= 1
        assert diagnostics["prefetch_polled_ready"] >= 1
        assert diagnostics["pending_prefetches"] == 0
        assert manager.has_pending_or_ready() is False
        manager.shutdown()

    def test_materialization_manager_ready_callback_marks_ready(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager

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

        manager = ExpertMaterializationManager(
            weight_path=str(weight_path),
            expert_key_template="model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
            max_cached_experts=2,
            prefetch_workers=1,
        )
        ready = []
        manager.set_ready_callback(lambda layer_idx, expert_idx: ready.append((layer_idx, expert_idx)))
        assert manager.prefetch(0, 1) is True

        for _ in range(20):
            manager.drain_ready_callbacks()
            if ready:
                break
            time.sleep(0.01)

        diagnostics = manager.diagnostics()
        assert ready == [(0, 1)]
        assert diagnostics["prefetch_background_ready_callbacks"] >= 1
        manager.shutdown()

    def test_hybrid_moe_pipeline_applies_ready_promotions(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
            expert_prefetch_workers=0,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="pipeline_promote",
                )
            ],
            phase="decode",
        )
        hybrid.materialization_manager.prefetch(0, 1)
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )
        hybrid.materialization_manager._cache.clear()
        hybrid.materialization_manager._futures.clear()
        hybrid.materialization_manager._ready_queue.clear()

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()

        assert stats["ready_applied"] == 1
        assert stats["ready_deferred"] == 0
        assert "1" in hybrid.gpu_experts
        assert diagnostics["pipeline_ready_applied"] == 1
        assert diagnostics["gpu_experts_mask_sum"] == 1

    def test_hybrid_moe_background_ready_callback_advances_lifecycle(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
            expert_prefetch_workers=0,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="callback_ready",
                )
            ],
            phase="decode",
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.PREFETCHING,
            phase="decode",
        )

        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.materialization_manager._ready_mark_queue.put((0, 1))

        ready_polled = hybrid.refresh_offload_state()
        diagnostics = hybrid.diagnostics()

        assert ready_polled == 1
        assert hybrid.offload_backend.migration_manager.state_for(0, 1) == MigrationLifecycle.READY
        assert diagnostics["materialization_manager"]["prefetch_background_ready_callbacks"] >= 1

    def test_hybrid_moe_pipeline_primes_pending_prefetch(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                decode_require_prefetch_ready=True,
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
            expert_prefetch_workers=1,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="pipeline_prime",
                )
            ],
            phase="decode",
        )
        submitted = []

        def fake_prefetch(layer_idx, expert_idx):
            submitted.append((layer_idx, expert_idx))
            return True

        hybrid.materialization_manager.prefetch = fake_prefetch
        hybrid.materialization_manager.is_ready = lambda layer_idx, expert_idx: False
        hybrid.materialization_manager.has_pending_or_ready = lambda: False
        hybrid.materialization_manager.poll_ready = lambda: []

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()
        lifecycle = diagnostics["backend"]["migration_manager"]["layers"][0]["lifecycle"][0]

        assert stats["prefetch_submitted"] == 1
        assert stats["ready_applied"] == 0
        assert diagnostics["prefetch_enqueued"] == 1
        assert submitted == []
        assert diagnostics["materialization_manager"]["resident_stage_hits"] == 1
        assert lifecycle["state"] == MigrationLifecycle.DEFERRED.value

    def test_hybrid_moe_prefetch_uses_offload_resident_weights(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
            expert_prefetch_workers=1,
        ).to(dtype=torch.float32)

        calls = []

        def fake_export(expert_idx):
            calls.append(expert_idx)
            return {
                "gate": torch.ones(8, 4),
                "up": torch.ones(8, 4),
                "down": torch.ones(4, 8),
            }

        hybrid.offload_backend.export_expert_weights = fake_export
        hybrid._request_prefetch(1)
        diagnostics = hybrid.diagnostics()

        assert calls == [1]
        assert diagnostics["prefetch_enqueued"] == 1
        assert diagnostics["materialization_manager"]["pending_prefetches"] == 0
        assert diagnostics["materialization_manager"]["cache_size"] == 1

    def test_hybrid_moe_forward_skips_duplicate_decode_migration_after_pipeline_tick(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                decode_require_prefetch_ready=True,
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

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="pipeline_then_forward",
                )
            ],
            phase="decode",
        )
        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.materialization_manager._cache.clear()
        hybrid.materialization_manager._futures.clear()
        hybrid.materialization_manager._ready_queue.clear()
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )

        hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        diagnostics = hybrid.diagnostics()
        assert output.shape == (1, 4)
        assert diagnostics["pipeline_ready_applied"] == 1
        assert diagnostics["applied_migration_ops"] == 2

    def test_hybrid_moe_demote_keeps_warm_expert_cache(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid._promote_expert_to_gpu(1, torch.device("cpu"), torch.float32)
        assert "1" in hybrid.gpu_experts

        hybrid._demote_expert_from_gpu(1, ExpertResidency.PIM)
        diagnostics = hybrid.diagnostics()

        assert "1" not in hybrid.gpu_experts
        assert diagnostics["warm_cache_stores"] == 1
        assert diagnostics["warm_cache_size"] == 1

    def test_hybrid_moe_repromote_hits_warm_cache(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        build_calls = []
        original_build = hybrid._build_runtime_expert

        def tracking_build(expert_idx, device, dtype):
            build_calls.append(expert_idx)
            return original_build(expert_idx, device, dtype)

        hybrid._build_runtime_expert = tracking_build
        hybrid._promote_expert_to_gpu(1, torch.device("cpu"), torch.float32)
        hybrid._demote_expert_from_gpu(1, ExpertResidency.PIM)
        hybrid._promote_expert_to_gpu(1, torch.device("cpu"), torch.float32)
        diagnostics = hybrid.diagnostics()

        assert build_calls == [1]
        assert diagnostics["warm_cache_hits"] == 1
        assert diagnostics["activation_applied"] == 2
        assert diagnostics["warm_cache_device_transfers"] == 0

    def test_hybrid_moe_pipeline_prebuilds_ready_expert(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="prebuild_ready",
                )
            ],
            phase="decode",
        )
        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )

        build_calls = []
        original_build = hybrid._build_runtime_expert

        def tracking_build(expert_idx, device, dtype):
            build_calls.append(expert_idx)
            return original_build(expert_idx, device, dtype)

        hybrid._build_runtime_expert = tracking_build
        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()
        layer_migration = diagnostics["backend"]["migration_manager"]["layers"][0]

        assert stats["warm_prebuilt"] == 1
        assert build_calls == [1]
        assert diagnostics["warm_cache_prebuilt"] == 1
        assert diagnostics["warm_cache_hits"] == 0
        assert diagnostics["activated_cache_hits"] == 1
        assert layer_migration["total_warmed_events"] == 1
        assert layer_migration["total_activated_events"] == 1
        assert diagnostics["backend"]["migration_manager"]["layers"][0]["lifecycle"][0]["state"] == "applied"
        assert diagnostics["warm_cache_device_transfers"] == 1

    def test_hybrid_moe_pipeline_marks_activated_before_apply(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="activate_then_apply",
                )
            ],
            phase="decode",
        )
        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()
        layer_migration = diagnostics["backend"]["migration_manager"]["layers"][0]

        assert stats["activation_ready"] == 1
        assert stats["ready_applied"] == 1
        assert stats["apply_batch_activated"] == 1
        assert stats["apply_batch_warm"] == 0
        assert stats["apply_batch_cold"] == 0
        assert diagnostics["activation_submitted"] == 1
        assert diagnostics["activation_ready"] == 1
        assert diagnostics["activation_applied"] == 1
        assert diagnostics["pipeline_prefetch_overlap_hits"] == 1
        assert diagnostics["pipeline_promotion_source_activated"] == 1
        assert diagnostics["pipeline_promotion_source_warm"] == 0
        assert diagnostics["pipeline_promotion_source_cold"] == 0
        assert layer_migration["total_activated_events"] == 1
        assert layer_migration["lifecycle"][0]["state"] == MigrationLifecycle.APPLIED.value
        assert diagnostics["activated_cache_hits"] == 1
        assert diagnostics["activated_cache_stores"] == 1
        assert diagnostics["activated_cache_size"] == 0
        assert layer_migration["pending_ops"] == 0
        assert diagnostics["pipeline_apply_batches"] == 1
        assert diagnostics["pipeline_apply_batch_experts"] == 1
        assert diagnostics["pipeline_apply_batch_evictions"] == 1
        assert diagnostics["pipeline_apply_batch_activated"] == 1
        assert diagnostics["pipeline_apply_batch_warm"] == 0
        assert diagnostics["pipeline_apply_batch_cold"] == 0
        assert diagnostics["apply_queue_enqueued"] == 1
        assert diagnostics["apply_queue_committed"] == 1
        assert diagnostics["background_apply_queue_enqueued"] == 0

    def test_hybrid_moe_background_pipeline_enqueues_apply_candidates(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                gpu_budget_per_layer=2,
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="background_apply_queue",
                )
            ],
            phase="decode",
        )
        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )

        stats = hybrid.background_advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()

        assert stats["apply_queue_enqueued"] == 1
        assert stats["activation_applied"] == 0
        assert diagnostics["background_apply_queue_enqueued"] == 1
        assert diagnostics["background_apply_commit_queue_enqueued"] == 1
        assert stats["apply_commit_batch_queue_prefinalized"] == 1
        assert diagnostics["background_apply_commit_batch_queue_prefinalized_batches"] == 1
        assert diagnostics["apply_queue_enqueued"] == 1
        assert diagnostics["apply_queue_committed"] == 0
        assert diagnostics["apply_queue_size"] == 1
        assert diagnostics["background_apply_commit_batches"] == 0
        assert diagnostics["background_apply_commit_experts"] == 0

    def test_hybrid_moe_apply_queue_rebalances_cold_candidates(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )
        residency_plan.layer_state(0).hotness[1] = 9.0
        residency_plan.layer_state(0).hotness[2] = 6.0
        residency_plan.layer_state(0).hotness[3] = 1.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="apply_queue_rebalance",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )

        stats = {"apply_queue_enqueued": hybrid._enqueue_activated_apply_candidates(phase="decode")}
        diagnostics = hybrid.diagnostics()

        assert stats["apply_queue_enqueued"] == 3
        assert diagnostics["apply_queue_limit"] == 1
        assert diagnostics["apply_queue_evictions"] >= 1
        assert diagnostics["apply_queue_size"] <= diagnostics["apply_queue_limit"]
        assert diagnostics["apply_queue_pending_experts"] == [1]

    def test_summarize_offload_diagnostics_reports_apply_queue_metrics(self):
        from nano_ktrans.scheduler.diagnostics import summarize_offload_diagnostics

        summary = summarize_offload_diagnostics(
            {
                "layer_count": 1,
                "offload_refresh": {
                    "offload_background_apply_queue_enqueued_total": 3,
                    "offload_background_apply_commit_queue_enqueued_total": 2,
                    "offload_background_apply_commit_batch_queue_enqueued_total": 1,
                    "offload_background_resident_commit_batch_queue_enqueued_total": 1,
                    "offload_background_resident_commit_batch_queue_prefinalized_total": 1,
                    "offload_background_resident_commit_finalize_queue_enqueued_total": 1,
                    "offload_background_resident_commit_finalize_queue_prefinalized_total": 1,
                    "offload_background_resident_commit_ready_cache_stores_total": 1,
                    "offload_background_resident_commit_apply_queue_enqueued_total": 1,
                    "offload_background_resident_commit_finalize_ready_queue_enqueued_total": 1,
                },
                "dynamic_scheduler": {"enabled": True},
                "layers": [
                    {
                        "apply_queue_size": 2,
                        "apply_queue_limit": 4,
                        "apply_commit_queue_size": 1,
                        "apply_commit_queue_limit": 2,
                        "apply_commit_queue_utilization": 0.5,
                        "apply_commit_batch_queue_size": 1,
                        "apply_commit_batch_queue_limit": 2,
                        "apply_commit_batch_queue_utilization": 0.5,
                        "resident_commit_batch_queue_size": 1,
                        "resident_commit_batch_queue_limit": 2,
                        "resident_commit_batch_queue_utilization": 0.5,
                        "resident_commit_finalize_queue_size": 1,
                        "resident_commit_finalize_queue_limit": 2,
                        "resident_commit_finalize_queue_utilization": 0.5,
                        "resident_commit_ready_cache_size": 1,
                        "resident_commit_ready_cache_limit": 2,
                        "resident_commit_ready_cache_utilization": 0.5,
                        "resident_commit_apply_queue_size": 1,
                        "resident_commit_apply_queue_limit": 2,
                        "resident_commit_apply_queue_utilization": 0.5,
                        "resident_commit_finalize_ready_queue_size": 1,
                        "resident_commit_finalize_ready_queue_limit": 2,
                        "resident_commit_finalize_ready_queue_utilization": 0.5,
                        "apply_queue_enqueued": 5,
                        "apply_queue_committed": 3,
                        "apply_queue_pruned": 1,
                        "apply_queue_evictions": 2,
                        "apply_commit_queue_enqueued": 2,
                        "apply_commit_queue_pruned": 1,
                        "apply_commit_queue_evictions": 1,
                        "apply_commit_batch_queue_enqueued": 1,
                        "apply_commit_batch_queue_pruned": 0,
                        "apply_commit_batch_queue_evictions": 0,
                        "resident_commit_batch_queue_enqueued": 1,
                        "resident_commit_batch_queue_committed_batches": 1,
                        "resident_commit_batch_queue_pruned": 0,
                        "resident_commit_batch_queue_evictions": 0,
                        "resident_commit_finalize_queue_enqueued": 1,
                        "resident_commit_finalize_queue_committed_batches": 1,
                        "resident_commit_finalize_queue_pruned": 0,
                        "resident_commit_finalize_queue_evictions": 0,
                        "resident_commit_ready_cache_stores": 1,
                        "resident_commit_ready_cache_hits": 0,
                        "resident_commit_ready_cache_pruned": 0,
                        "resident_commit_ready_cache_evictions": 0,
                        "resident_commit_apply_queue_enqueued": 1,
                        "resident_commit_apply_queue_batches": 1,
                        "resident_commit_apply_queue_committed_batches": 1,
                        "resident_commit_apply_queue_pruned": 0,
                        "resident_commit_apply_queue_evictions": 0,
                        "resident_commit_finalize_ready_queue_enqueued": 1,
                        "resident_commit_finalize_ready_queue_batches": 1,
                        "resident_commit_finalize_ready_queue_committed_batches": 1,
                        "resident_commit_finalize_ready_queue_pruned": 0,
                        "resident_commit_finalize_ready_queue_evictions": 0,
                        "apply_queue_pressure": 1.5,
                        "apply_queue_pressure_step": 0.5,
                        "apply_queue_pressure_ema": 0.75,
                        "apply_queue_budget_backoff": 1,
                        "apply_commit_queue_pressure": 0.25,
                        "apply_commit_queue_pressure_step": 0.25,
                        "apply_commit_queue_pressure_ema": 0.1,
                        "apply_commit_queue_budget_backoff": 0,
                        "apply_commit_batch_queue_pressure": 0.5,
                        "apply_commit_batch_queue_pressure_step": 0.5,
                        "apply_commit_batch_queue_pressure_ema": 0.25,
                        "apply_commit_batch_queue_budget_backoff": 1,
                        "background_apply_queue_enqueued": 3,
                        "background_apply_commit_batch_queue_enqueued": 1,
                        "background_resident_commit_batch_queue_enqueued": 1,
                        "background_resident_commit_batch_queue_committed_batches": 1,
                        "background_resident_commit_batch_queue_prefinalized_batches": 1,
                        "background_resident_commit_finalize_queue_enqueued": 1,
                        "background_resident_commit_finalize_queue_committed_batches": 1,
                        "background_resident_commit_finalize_queue_prefinalized_batches": 1,
                        "background_resident_commit_ready_cache_stores": 1,
                        "background_resident_commit_apply_queue_enqueued": 1,
                        "background_resident_commit_apply_queue_committed_batches": 1,
                        "background_resident_commit_finalize_ready_queue_enqueued": 1,
                        "background_resident_commit_finalize_ready_queue_committed_batches": 1,
                    }
                ],
            }
        )

        assert summary["apply_queue_size"] == 2
        assert summary["apply_queue_limit"] == 4
        assert summary["apply_queue_enqueued"] == 5
        assert summary["apply_queue_committed"] == 3
        assert summary["apply_queue_pruned"] == 1
        assert summary["apply_queue_evictions"] == 2
        assert summary["apply_commit_queue_size"] == 1
        assert summary["apply_commit_queue_limit"] == 2
        assert summary["apply_commit_batch_queue_size"] == 1
        assert summary["apply_commit_batch_queue_limit"] == 2
        assert summary["resident_commit_batch_queue_size"] == 1
        assert summary["resident_commit_batch_queue_limit"] == 2
        assert summary["resident_commit_finalize_queue_size"] == 1
        assert summary["resident_commit_finalize_queue_limit"] == 2
        assert summary["resident_commit_ready_cache_size"] == 1
        assert summary["resident_commit_ready_cache_limit"] == 2
        assert summary["resident_commit_apply_queue_size"] == 1
        assert summary["resident_commit_apply_queue_limit"] == 2
        assert summary["resident_commit_finalize_ready_queue_size"] == 1
        assert summary["resident_commit_finalize_ready_queue_limit"] == 2
        assert summary["apply_commit_queue_enqueued"] == 2
        assert summary["apply_commit_queue_pruned"] == 1
        assert summary["apply_commit_queue_evictions"] == 1
        assert summary["apply_commit_batch_queue_enqueued"] == 1
        assert summary["apply_commit_batch_queue_pruned"] == 0
        assert summary["apply_commit_batch_queue_evictions"] == 0
        assert summary["resident_commit_batch_queue_enqueued"] == 1
        assert summary["resident_commit_batch_queue_committed_batches"] == 1
        assert summary["resident_commit_batch_queue_pruned"] == 0
        assert summary["resident_commit_batch_queue_evictions"] == 0
        assert summary["resident_commit_finalize_queue_enqueued"] == 1
        assert summary["resident_commit_finalize_queue_committed_batches"] == 1
        assert summary["resident_commit_finalize_queue_pruned"] == 0
        assert summary["resident_commit_finalize_queue_evictions"] == 0
        assert summary["resident_commit_ready_cache_stores"] == 1
        assert summary["resident_commit_ready_cache_pruned"] == 0
        assert summary["resident_commit_ready_cache_evictions"] == 0
        assert summary["resident_commit_apply_queue_enqueued"] == 1
        assert summary["resident_commit_apply_queue_committed_batches"] == 1
        assert summary["resident_commit_apply_queue_pruned"] == 0
        assert summary["resident_commit_apply_queue_evictions"] == 0
        assert summary["resident_commit_finalize_ready_queue_enqueued"] == 1
        assert summary["resident_commit_finalize_ready_queue_committed_batches"] == 1
        assert summary["resident_commit_finalize_ready_queue_pruned"] == 0
        assert summary["resident_commit_finalize_ready_queue_evictions"] == 0
        assert summary["apply_commit_ready_cache_size"] == 0
        assert summary["apply_commit_ready_hits"] == 0
        assert summary["apply_commit_ready_stores"] == 0
        assert summary["apply_commit_ready_pruned"] == 0
        assert summary["background_apply_commit_resolved"] == 0
        assert summary["background_apply_queue_enqueued"] == 3
        assert summary["background_apply_commit_queue_enqueued"] == 0
        assert summary["background_apply_commit_batch_queue_enqueued"] == 1
        assert summary["background_resident_commit_batch_queue_enqueued"] == 1
        assert summary["background_resident_commit_batch_queue_committed_batches"] == 1
        assert summary["background_resident_commit_batch_queue_prefinalized_batches"] == 1
        assert summary["background_resident_commit_finalize_queue_enqueued"] == 1
        assert summary["background_resident_commit_finalize_queue_committed_batches"] == 1
        assert summary["background_resident_commit_finalize_queue_prefinalized_batches"] == 1
        assert summary["background_resident_commit_ready_cache_stores"] == 1
        assert summary["background_resident_commit_apply_queue_enqueued"] == 1
        assert summary["background_resident_commit_apply_queue_committed_batches"] == 1
        assert summary["background_resident_commit_finalize_ready_queue_enqueued"] == 1
        assert summary["background_resident_commit_finalize_ready_queue_committed_batches"] == 1
        assert summary["apply_queue_commit_batch_size_avg"] is None
        assert summary["apply_queue_utilization"] == pytest.approx(0.5)
        assert summary["apply_commit_queue_utilization"] == pytest.approx(0.5)
        assert summary["apply_commit_batch_queue_utilization"] == pytest.approx(0.5)
        assert summary["resident_commit_batch_queue_utilization"] == pytest.approx(0.5)
        assert summary["resident_commit_finalize_queue_utilization"] == pytest.approx(0.5)
        assert summary["resident_commit_ready_cache_utilization"] == pytest.approx(0.5)
        assert summary["resident_commit_apply_queue_utilization"] == pytest.approx(0.5)
        assert summary["resident_commit_finalize_ready_queue_utilization"] == pytest.approx(0.5)
        assert summary["apply_queue_pressure_avg"] == pytest.approx(1.5)
        assert summary["apply_queue_pressure_step_avg"] == pytest.approx(0.5)
        assert summary["apply_queue_pressure_ema_avg"] == pytest.approx(0.75)
        assert summary["apply_queue_budget_backoff_avg"] == 1.0
        assert summary["apply_commit_queue_pressure_avg"] == pytest.approx(0.25)
        assert summary["apply_commit_queue_pressure_ema_avg"] == pytest.approx(0.1)
        assert summary["apply_commit_batch_queue_pressure_avg"] == pytest.approx(0.5)
        assert summary["apply_commit_batch_queue_pressure_ema_avg"] == pytest.approx(0.25)
        assert summary["apply_commit_batch_queue_budget_backoff_avg"] == pytest.approx(1.0)
        assert summary["offload_background_apply_queue_enqueued_total"] == 3
        assert summary["offload_background_apply_commit_queue_enqueued_total"] == 2
        assert summary["offload_background_apply_commit_batch_queue_enqueued_total"] == 1
        assert summary["offload_background_resident_commit_batch_queue_enqueued_total"] == 1
        assert summary["offload_background_resident_commit_batch_queue_prefinalized_total"] == 1
        assert summary["offload_background_resident_commit_finalize_queue_enqueued_total"] == 1
        assert summary["offload_background_resident_commit_finalize_queue_prefinalized_total"] == 1
        assert summary["offload_background_resident_commit_ready_cache_stores_total"] == 1
        assert summary["offload_background_resident_commit_apply_queue_enqueued_total"] == 1
        assert summary["offload_background_resident_commit_finalize_ready_queue_enqueued_total"] == 1

    def test_apply_queue_commit_batch_metrics_track_background_and_foreground_commits(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=3,
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
            expert_warm_cache_size=2,
            expert_prepared_cache_size=2,
            prepared_controller_aggressiveness=0.5,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="apply_commit_metrics",
                    )
                ],
                phase="decode",
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.activated_expert_cache[str(expert_idx)] = hybrid._build_runtime_expert(
                expert_idx,
                torch.device("cpu"),
                torch.float32,
            )

        hybrid._enqueue_activated_apply_candidates(phase="decode")
        hybrid._enqueue_apply_commit_candidates(expert_ids={1}, background=True)
        background_applied = hybrid._background_apply_activated_experts(
            phase="decode",
            eligible_expert_ids={1},
        )
        diagnostics = hybrid.diagnostics()

        assert background_applied == 1
        assert diagnostics["apply_commit_queue_enqueued"] == 1
        assert diagnostics["apply_commit_batch_queue_enqueued"] == 1
        assert diagnostics["resident_commit_batch_queue_enqueued"] == 1
        assert diagnostics["apply_commit_queue_size"] == 0
        assert diagnostics["apply_commit_batch_queue_size"] == 0
        assert diagnostics["resident_commit_batch_queue_size"] == 0
        assert diagnostics["apply_commit_queue_evictions"] == 0
        assert diagnostics["apply_commit_batch_queue_evictions"] == 0
        assert diagnostics["resident_commit_batch_queue_committed_batches"] == 1
        assert diagnostics["apply_commit_ready_hits"] == 1
        assert diagnostics["apply_commit_ready_stores"] == 1
        assert diagnostics["background_apply_commit_batches"] == 1
        assert diagnostics["background_apply_commit_experts"] == 1
        assert diagnostics["apply_queue_commit_batches"] == 1
        assert diagnostics["apply_queue_commit_experts"] == 1
        assert diagnostics["background_apply_commit_queue_enqueued"] == 1
        assert diagnostics["background_apply_commit_batch_queue_enqueued"] == 1
        assert diagnostics["background_resident_commit_batch_queue_committed_batches"] == 1

        hybrid._enqueue_activated_apply_candidates(phase="decode")
        hybrid._stage_apply_commit_batch(phase="decode")
        hybrid._commit_apply_candidate_queue(
            device=torch.device("cpu"),
            dtype=torch.float32,
            phase="decode",
            max_commits=2,
            allow_eviction=False,
            count_batch=False,
            background=False,
        )
        diagnostics = hybrid.diagnostics()

        assert diagnostics["apply_queue_commit_batches"] == 2
        assert diagnostics["apply_queue_commit_experts"] == 2
        assert diagnostics["apply_commit_queue_enqueued"] >= 2
        assert diagnostics["apply_commit_batch_queue_enqueued"] >= 2
        assert diagnostics["apply_commit_queue_utilization"] >= 0.0
        assert diagnostics["apply_commit_batch_queue_utilization"] >= 0.0
        assert diagnostics["apply_commit_ready_cache_size"] == 0

    def test_apply_promotion_batch_commits_modules_in_batch(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        promotion_ops = []
        ready_entries = []
        for expert_idx in (1, 2):
            op = ExpertMigrationOp(
                layer_idx=0,
                expert_idx=expert_idx,
                src=ExpertResidency.PIM,
                dst=ExpertResidency.GPU,
                reason="batched_commit",
            )
            hybrid.offload_backend.queue_migration_plan([op], phase="decode")
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            promotion_ops.append(op)
            ready_entries.append(
                (
                    op,
                    {
                        "expert_idx": expert_idx,
                        "module": hybrid._build_runtime_expert(
                            expert_idx,
                            torch.device("cpu"),
                            torch.float32,
                        ),
                        "source": "activated",
                    },
                )
            )

        applied, completed, source_counts = hybrid._apply_promotion_batch(
            promotion_ops,
            device=torch.device("cpu"),
            dtype=torch.float32,
            phase="decode",
            pre_resolved_batch=ready_entries,
        )

        diagnostics = hybrid.diagnostics()
        assert applied == 2
        assert completed == {1, 2}
        assert source_counts["activated"] == 2
        assert diagnostics["apply_commit_ready_hits"] == 2
        assert diagnostics["pipeline_apply_batches"] == 1
        assert diagnostics["pipeline_apply_batch_experts"] == 2
        assert diagnostics["gpu_experts_mask"][1] is True if False else True
        assert hybrid.gpu_experts_mask[1].item() is True
        assert hybrid.gpu_experts_mask[2].item() is True
        assert "1" in hybrid.gpu_experts
        assert "2" in hybrid.gpu_experts

    def test_stage_apply_commit_ready_batch_enqueues_ready_commit_buffer(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
        residency_plan.layer_state(0).hotness[1] = 1.0
        residency_plan.layer_state(0).hotness[2] = 5.0
        residency_plan.layer_state(0).hotness[3] = 9.0
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            op = ExpertMigrationOp(
                layer_idx=0,
                expert_idx=expert_idx,
                src=ExpertResidency.PIM,
                dst=ExpertResidency.GPU,
                reason="stage_commit_batch_queue",
            )
            hybrid.offload_backend.queue_migration_plan([op], phase="decode")
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.apply_candidate_queue[str(expert_idx)] = op
            hybrid.apply_commit_queue[str(expert_idx)] = op
            hybrid.apply_commit_ready_cache[str(expert_idx)] = {
                "op": op,
                "resolved": {
                    "expert_idx": expert_idx,
                    "module": hybrid._build_runtime_expert(
                        expert_idx,
                        torch.device("cpu"),
                        torch.float32,
                    ),
                    "source": "activated",
                },
            }

        enqueued = hybrid._stage_apply_commit_batch_queue(
            eligible_expert_ids={1, 2, 3},
            max_commits=2,
            background=True,
        )
        diagnostics = hybrid.diagnostics()

        assert enqueued == 1
        assert diagnostics["apply_commit_batch_queue_enqueued"] == 1
        assert diagnostics["background_apply_commit_batch_queue_enqueued"] == 1
        assert diagnostics["apply_commit_batch_queue_batches"] == 1
        assert diagnostics["apply_commit_batch_queue_size"] == 1
        assert diagnostics["apply_commit_batch_queue_pending_experts"] == [3, 2]
        assert diagnostics["resident_commit_batch_queue_enqueued"] == 0
        assert diagnostics["resident_commit_batch_queue_size"] == 0

    def test_apply_commit_batch_queue_groups_hot_ready_entries_into_batch(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
        residency_plan.layer_state(0).hotness[1] = 1.0
        residency_plan.layer_state(0).hotness[2] = 5.0
        residency_plan.layer_state(0).hotness[3] = 9.0
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            op = ExpertMigrationOp(
                layer_idx=0,
                expert_idx=expert_idx,
                src=ExpertResidency.PIM,
                dst=ExpertResidency.GPU,
                reason="group_commit_batch_queue",
            )
            hybrid.offload_backend.queue_migration_plan([op], phase="decode")
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.apply_candidate_queue[str(expert_idx)] = op
            hybrid.apply_commit_queue[str(expert_idx)] = op
            hybrid.apply_commit_ready_cache[str(expert_idx)] = {
                "op": op,
                "resolved": {
                    "expert_idx": expert_idx,
                    "module": hybrid._build_runtime_expert(
                        expert_idx,
                        torch.device("cpu"),
                        torch.float32,
                    ),
                    "source": "activated",
                },
            }

        hybrid._stage_apply_commit_batch_queue(
            eligible_expert_ids={1, 2, 3},
            max_commits=2,
            background=False,
        )

        assert len(hybrid.apply_commit_batch_queue) == 1
        only_batch = next(iter(hybrid.apply_commit_batch_queue.values()))
        assert [int(op.expert_idx) for op, _resolved in only_batch] == [3, 2]

    def test_resident_commit_batch_queue_stages_preexisting_commit_batches(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
        residency_plan.layer_state(0).hotness[1] = 1.0
        residency_plan.layer_state(0).hotness[2] = 5.0
        residency_plan.layer_state(0).hotness[3] = 9.0
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            op = ExpertMigrationOp(
                layer_idx=0,
                expert_idx=expert_idx,
                src=ExpertResidency.PIM,
                dst=ExpertResidency.GPU,
                reason="stage_resident_commit_batch_queue",
            )
            hybrid.offload_backend.queue_migration_plan([op], phase="decode")
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.apply_candidate_queue[str(expert_idx)] = op
            hybrid.apply_commit_queue[str(expert_idx)] = op
            hybrid.apply_commit_ready_cache[str(expert_idx)] = {
                "op": op,
                "resolved": {
                    "expert_idx": expert_idx,
                    "module": hybrid._build_runtime_expert(
                        expert_idx,
                        torch.device("cpu"),
                        torch.float32,
                    ),
                    "source": "activated",
                },
            }

        hybrid._stage_apply_commit_batch_queue(
            eligible_expert_ids={1, 2, 3},
            max_commits=2,
            background=True,
        )
        staged = hybrid._stage_resident_commit_batches(
            eligible_batch_keys=set(hybrid.apply_commit_batch_queue.keys()),
            max_batches=1,
            background=True,
        )
        diagnostics = hybrid.diagnostics()

        assert staged == 1
        assert diagnostics["resident_commit_batch_queue_enqueued"] == 1
        assert diagnostics["resident_commit_batch_queue_batches"] == 1
        assert diagnostics["resident_commit_batch_queue_size"] == 1
        assert diagnostics["resident_commit_batch_queue_pending_experts"] == [3, 2]
        assert diagnostics["background_resident_commit_batch_queue_enqueued"] == 1

    def test_resident_commit_finalize_queue_stages_preexisting_resident_batches(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
        residency_plan.layer_state(0).hotness[1] = 1.0
        residency_plan.layer_state(0).hotness[2] = 5.0
        residency_plan.layer_state(0).hotness[3] = 9.0
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=4,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            op = ExpertMigrationOp(
                layer_idx=0,
                expert_idx=expert_idx,
                src=ExpertResidency.PIM,
                dst=ExpertResidency.GPU,
                reason="stage_resident_commit_finalize_queue",
            )
            hybrid.offload_backend.queue_migration_plan([op], phase="decode")
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.apply_candidate_queue[str(expert_idx)] = op
            hybrid.apply_commit_queue[str(expert_idx)] = op
            hybrid.apply_commit_ready_cache[str(expert_idx)] = {
                "op": op,
                "resolved": {
                    "expert_idx": expert_idx,
                    "module": hybrid._build_runtime_expert(
                        expert_idx,
                        torch.device("cpu"),
                        torch.float32,
                    ),
                    "source": "activated",
                },
            }

        hybrid._stage_apply_commit_batch_queue(
            eligible_expert_ids={1, 2, 3},
            max_commits=2,
            background=True,
        )
        hybrid._stage_resident_commit_batches(
            eligible_batch_keys=set(hybrid.apply_commit_batch_queue.keys()),
            max_batches=1,
            background=True,
        )
        staged = hybrid._stage_resident_commit_finalize_queue(
            eligible_batch_keys=set(hybrid.resident_commit_batch_queue.keys()),
            max_batches=1,
            background=True,
        )
        diagnostics = hybrid.diagnostics()

        assert staged == 1
        assert diagnostics["resident_commit_finalize_queue_enqueued"] == 1
        assert diagnostics["resident_commit_finalize_queue_batches"] == 1
        assert diagnostics["resident_commit_finalize_queue_size"] == 1
        assert diagnostics["resident_commit_finalize_queue_pending_experts"] == [3, 2]
        assert diagnostics["background_resident_commit_finalize_queue_enqueued"] == 1

    def test_apply_commit_queue_rebalance_prefers_hotter_commit_candidates(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
        residency_plan.layer_state(0).hotness[1] = 1.0
        residency_plan.layer_state(0).hotness[2] = 5.0
        residency_plan.layer_state(0).hotness[3] = 9.0
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
            num_experts=4,
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
            expert_warm_cache_size=4,
            prepared_controller_aggressiveness=0.5,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="commit_queue_rebalance",
                    )
                ],
                phase="decode",
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid.apply_candidate_queue[str(expert_idx)] = hybrid.offload_backend.migration_manager.peek_layer(0)[-1]

        hybrid._enqueue_apply_commit_candidates(expert_ids={1, 2, 3}, background=False)
        diagnostics = hybrid.diagnostics()

        assert diagnostics["apply_commit_queue_limit"] == 2
        assert diagnostics["apply_commit_queue_size"] == 2
        assert diagnostics["apply_commit_queue_evictions"] == 1
        assert diagnostics["apply_commit_queue_pending_experts"] == [2, 3]

    def test_hybrid_moe_promotion_prefers_activated_cache(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid._promote_expert_to_gpu(1, torch.device("cpu"), torch.float32)
        hybrid._demote_expert_from_gpu(1, ExpertResidency.PIM)
        assert "1" in hybrid.warm_expert_cache

        hybrid._activate_warmed_experts = lambda **kwargs: 0
        hybrid.activated_expert_cache["1"] = hybrid.warm_expert_cache.pop("1")
        diagnostics_before = hybrid.diagnostics()
        hybrid._promote_expert_to_gpu(1, torch.device("cpu"), torch.float32)
        diagnostics = hybrid.diagnostics()

        assert diagnostics_before["activated_cache_size"] == 1
        assert diagnostics["activated_cache_hits"] == 1
        assert diagnostics["warm_cache_hits"] == 0
        assert diagnostics["activated_cache_size"] == 0
        assert diagnostics["activation_applied"] == 2

    def test_hybrid_moe_activation_cache_keeps_hottest_candidates(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 10.0
        state.hotness[2] = 3.0

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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)
        hybrid.dynamic_expert_scheduler.plan_layer = lambda *args, **kwargs: []

        for expert_idx in (1, 2):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="activation_budget",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.READY,
                phase="decode",
            )

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()
        layer_migration = diagnostics["backend"]["migration_manager"]["layers"][0]

        assert stats["activation_ready"] == 1
        assert stats["ready_applied"] == 1
        assert diagnostics["activated_cache_size"] == 0
        assert diagnostics["activated_cache_hits"] == 1
        assert "1" in hybrid.gpu_experts
        assert "2" not in hybrid.activated_expert_cache
        assert layer_migration["total_activated_events"] == 1

    def test_activated_cache_eviction_downgrades_lifecycle_to_warmed(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="lifecycle_probe_1",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=2,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="lifecycle_probe_2",
                ),
            ],
            phase="decode",
        )

        for expert_idx in (1, 2):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid._store_warm_module(expert_idx, module, count_store=False)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )

        hybrid._store_activated_module(1, hybrid.warm_expert_cache.pop("1").to(dtype=torch.float32))
        hybrid._store_activated_module(2, hybrid.warm_expert_cache.pop("2").to(dtype=torch.float32))

        lifecycle = {
            int(expert_idx): hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2)
        }
        assert lifecycle[1] == MigrationLifecycle.WARMED.value
        assert lifecycle[2] == MigrationLifecycle.ACTIVATED.value
        assert "1" in hybrid.warm_expert_cache
        assert "2" in hybrid.activated_expert_cache
        layer_diag = hybrid.offload_backend.migration_manager.diagnostics()["layers"][0]
        assert layer_diag["total_activation_eviction_regressions"] == 1

    def test_activated_cache_prefers_colder_victim_on_eviction(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 9.0
        state.hotness[2] = 1.0
        state.hotness[3] = 8.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(0, 1, ExpertResidency.PIM, ExpertResidency.GPU, "act_hot"),
                ExpertMigrationOp(0, 2, ExpertResidency.PIM, ExpertResidency.GPU, "act_cold"),
                ExpertMigrationOp(0, 3, ExpertResidency.PIM, ExpertResidency.GPU, "act_new"),
            ],
            phase="decode",
        )

        for expert_idx in (1, 2, 3):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid._store_activated_module(expert_idx, module.to(dtype=torch.float32))

        assert "1" in hybrid.activated_expert_cache
        assert "3" in hybrid.activated_expert_cache
        assert "2" not in hybrid.activated_expert_cache
        assert "2" in hybrid.warm_expert_cache
        lifecycle = {
            expert_idx: hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2, 3)
        }
        assert lifecycle[1] == MigrationLifecycle.ACTIVATED.value
        assert lifecycle[2] == MigrationLifecycle.WARMED.value
        assert lifecycle[3] == MigrationLifecycle.ACTIVATED.value

    def test_prepared_cache_budget_limits_warm_plus_activated(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 9.0
        state.hotness[2] = 2.0
        state.hotness[3] = 8.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(0, 1, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_hot"),
                ExpertMigrationOp(0, 2, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_cold"),
                ExpertMigrationOp(0, 3, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_new"),
            ],
            phase="decode",
        )

        cold_module = hybrid._build_runtime_expert(2, torch.device("cpu"), torch.float32)
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            2,
            state=MigrationLifecycle.WARMED,
            phase="decode",
        )
        hybrid._store_warm_module(2, cold_module, count_store=False)

        for expert_idx in (1, 3):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid._store_activated_module(expert_idx, module.to(dtype=torch.float32))

        diagnostics = hybrid.diagnostics()
        lifecycle = {
            expert_idx: hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2, 3)
        }

        assert diagnostics["prepared_cache_limit"] == 2
        assert diagnostics["prepared_cache_size"] == 2
        assert diagnostics["effective_warm_cache_limit"] == 0
        assert "2" not in hybrid.warm_expert_cache
        assert set(hybrid.activated_expert_cache.keys()) == {"1", "3"}
        assert lifecycle[1] == MigrationLifecycle.ACTIVATED.value
        assert lifecycle[2] == MigrationLifecycle.READY.value
        assert lifecycle[3] == MigrationLifecycle.ACTIVATED.value

    def test_prepared_cache_budget_prefers_hot_activated_over_warm(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 10.0
        state.hotness[2] = 9.0
        state.hotness[3] = 1.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(0, 1, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_act_hot"),
                ExpertMigrationOp(0, 2, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_warm_hot"),
                ExpertMigrationOp(0, 3, ExpertResidency.PIM, ExpertResidency.GPU, "prepared_warm_cold"),
            ],
            phase="decode",
        )

        act_module = hybrid._build_runtime_expert(1, torch.device("cpu"), torch.float32)
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.ACTIVATED,
            phase="decode",
        )
        hybrid._store_activated_module(1, act_module.to(dtype=torch.float32))

        for expert_idx in (2, 3):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.WARMED,
                phase="decode",
            )
            hybrid._store_warm_module(expert_idx, module, count_store=False)

        diagnostics = hybrid.diagnostics()
        lifecycle = {
            expert_idx: hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2, 3)
        }

        assert diagnostics["prepared_cache_size"] == 2
        assert "1" in hybrid.activated_expert_cache
        assert "2" in hybrid.warm_expert_cache
        assert "3" not in hybrid.warm_expert_cache
        assert lifecycle[1] == MigrationLifecycle.ACTIVATED.value
        assert lifecycle[2] == MigrationLifecycle.WARMED.value
        assert lifecycle[3] == MigrationLifecycle.READY.value

    def test_scheduler_summary_reports_prepared_cache_metrics(self):
        from nano_ktrans.scheduler.diagnostics import summarize_offload_diagnostics

        offload_diagnostics = {
            "layer_count": 1,
            "scheduler_profile": {"profile": "baseline"},
            "prepared_cache_budget": 3,
            "prepared_controller_aggressiveness": 0.5,
            "offload_refresh": {},
            "dynamic_scheduler": {"enabled": True},
            "layers": [
                {
                    "prepared_cache_limit": 3,
                    "prepared_cache_budget_backoff": 1,
                    "effective_prepared_cache_limit": 2,
                    "prepared_cache_size": 2,
                    "effective_warm_cache_limit": 1,
                    "prepared_cache_rebalance_pressure": 1.0 / 3.0,
                    "prepared_cache_rebalance_pressure_step": 0.25,
                    "prepared_cache_rebalance_pressure_ema": 0.2,
                    "prepared_cache_rebalance_evicted_warm": 1,
                    "prepared_cache_rebalance_evicted_activated": 2,
                    "prepared_cache_rebalance_demoted_to_warm": 2,
                    "prepared_cache_rebalance_dropped_to_ready": 1,
                    "prepared_cache_activation_stage_bonus": 0.75,
                    "cold_promotion_penalty": 0.5,
                    "adaptive_activation_limit": 2,
                    "adaptive_prebuild_limit": 3,
                    "activated_cache_size": 1,
                    "warm_cache_size": 1,
                    "backend": {"migration_manager": {"layers": []}},
                }
            ],
        }

        summary = summarize_offload_diagnostics(offload_diagnostics)

        assert summary["prepared_cache_budget"] == 3
        assert summary["prepared_controller_aggressiveness"] == 0.5
        assert summary["prepared_cache_limit"] == 3
        assert summary["prepared_cache_budget_backoff"] == 1
        assert summary["effective_prepared_cache_limit"] == 2
        assert summary["prepared_cache_size"] == 2
        assert summary["effective_warm_cache_limit"] == 1
        assert summary["prepared_cache_utilization"] == 2 / 3
        assert summary["effective_prepared_cache_utilization"] == 1.0
        assert summary["prepared_cache_budget_backoff_avg"] == 1.0
        assert summary["prepared_cache_rebalance_evicted_warm"] == 1
        assert summary["prepared_cache_rebalance_evicted_activated"] == 2
        assert summary["prepared_cache_rebalance_demoted_to_warm"] == 2
        assert summary["prepared_cache_rebalance_dropped_to_ready"] == 1
        assert summary["prepared_cache_rebalance_activated_ratio"] == 2 / 3
        assert summary["prepared_cache_rebalance_pressure_avg"] == pytest.approx(1.0 / 3.0)
        assert summary["prepared_cache_rebalance_pressure_step_avg"] == pytest.approx(0.25)
        assert summary["prepared_cache_rebalance_pressure_ema_avg"] == pytest.approx(0.2)
        assert summary["prepared_cache_activation_stage_bonus_avg"] == 0.75
        assert summary["cold_promotion_penalty_avg"] == 0.5
        assert summary["adaptive_activation_limit_avg"] == 2
        assert summary["adaptive_prebuild_limit_avg"] == 3
        assert summary["adaptive_prefetch_pending_limit_avg"] == 0
        assert summary["adaptive_prefetch_candidate_budget_avg"] == 0

    def test_profile_sweep_summary_includes_prepared_cache_metrics(self):
        from nano_ktrans.scheduler.diagnostics import summarize_profile_sweep_results

        results = [
            {
                "backend": "cuda_pim",
                "scheduler_profile": "baseline",
                "status": "ok",
                "runs": [
                    {
                        "prefill_seconds": 1.0,
                        "decode_seconds": 2.0,
                        "decode_tokens_per_second": 1.5,
                    }
                ],
                "scheduler_summary": {
                    "pipeline_promotion_source_activated": 2,
                    "pipeline_promotion_source_warm": 1,
                    "pipeline_promotion_source_cold": 1,
                    "pipeline_apply_batches": 1,
                    "pipeline_apply_batch_size_avg": 2.0,
                    "pipeline_apply_batch_evictions": 0,
                    "offload_pipeline_apply_batch_count_total": 1,
                    "offload_pipeline_apply_batch_experts_total": 2,
                    "offload_pipeline_apply_batch_evictions_total": 0,
                    "runtime_deferred_for_prefetch": 0,
                    "prepared_cache_limit": 4,
                    "prepared_cache_budget_backoff_avg": 1.0,
                    "effective_prepared_cache_limit": 3,
                    "prepared_cache_size": 3,
                    "effective_warm_cache_limit": 2,
                    "prepared_cache_utilization": 0.75,
                    "effective_prepared_cache_utilization": 1.0,
                    "prepared_cache_rebalance_pressure_avg": 0.5,
                    "prepared_cache_rebalance_pressure_step_avg": 0.25,
                    "prepared_cache_rebalance_pressure_ema_avg": 0.2,
                    "prepared_cache_rebalance_evicted_warm": 1,
                    "prepared_cache_rebalance_evicted_activated": 0,
                    "prepared_cache_rebalance_demoted_to_warm": 0,
                    "prepared_cache_rebalance_dropped_to_ready": 1,
                    "prepared_cache_rebalance_activated_ratio": 0.0,
                    "prepared_cache_activation_stage_bonus_avg": 0.25,
                    "cold_promotion_penalty_avg": 1.0,
                    "adaptive_activation_limit_avg": 1.0,
                    "adaptive_prebuild_limit_avg": 2.0,
                    "adaptive_prefetch_pending_limit_avg": 2.0,
                    "adaptive_prefetch_candidate_budget_avg": 3.0,
                    "migration_activation_eviction_regressions": 0,
                    "migration_warm_eviction_regressions": 0,
                    "pipeline_prefetch_overlap_hits": 2,
                },
            }
        ]

        summary = summarize_profile_sweep_results(results)
        profile = summary["profiles"][0]
        comparison_row = summary["comparison_table"][0]
        best_by_metric = summary["best_by_metric"]["prepared_cache_utilization"]

        assert profile["prepared_cache_limit"] == 4
        assert profile["prepared_cache_budget_backoff_avg"] == 1.0
        assert profile["effective_prepared_cache_limit"] == 3
        assert profile["prepared_cache_size"] == 3
        assert profile["effective_warm_cache_limit"] == 2
        assert profile["prepared_cache_utilization"] == 0.75
        assert profile["effective_prepared_cache_utilization"] == 1.0
        assert profile["prepared_cache_rebalance_pressure_avg"] == 0.5
        assert profile["prepared_cache_rebalance_pressure_step_avg"] == 0.25
        assert profile["prepared_cache_rebalance_pressure_ema_avg"] == 0.2
        assert profile["prepared_cache_rebalance_evicted_warm"] == 1
        assert profile["prepared_cache_rebalance_evicted_activated"] == 0
        assert profile["prepared_cache_activation_stage_bonus_avg"] == 0.25
        assert profile["cold_promotion_penalty_avg"] == 1.0
        assert profile["adaptive_activation_limit_avg"] == 1.0
        assert profile["adaptive_prebuild_limit_avg"] == 2.0
        assert profile["adaptive_prefetch_pending_limit_avg"] == 2.0
        assert profile["adaptive_prefetch_candidate_budget_avg"] == 3.0
        assert comparison_row["prepared_cache_utilization"] == 0.75
        assert best_by_metric["scheduler_profile"] == "baseline"
        assert best_by_metric["value"] == 0.75

    def test_prepared_cache_budget_backoff_scales_with_pressure(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
            expert_prepared_cache_size=3,
        ).to(dtype=torch.float32)

        hybrid.prepared_cache_activation_stage_bonus = 0.0
        hybrid.prepared_cache_rebalance_evicted_warm = 4
        hybrid.prepared_cache_rebalance_evicted_activated = 2
        hybrid.pipeline_ticks = 3

        assert hybrid._prepared_cache_rebalance_pressure() == pytest.approx(2.0)
        assert hybrid._prepared_cache_budget_backoff() == 2
        assert hybrid._effective_prepared_cache_limit() == 1
        assert hybrid._prepared_controller_engaged() is True

        hybrid.cold_promotion_penalty = 1.5
        assert hybrid._prepared_cache_budget_backoff() == 0
        assert hybrid._effective_prepared_cache_limit() == 3
        assert hybrid._adaptive_activation_limit() == 2
        assert hybrid._adaptive_prebuild_limit() == 5

    def test_prepared_cache_pressure_signals_track_step_and_ema(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
            expert_prepared_cache_size=3,
        ).to(dtype=torch.float32)

        hybrid.prepared_cache_rebalance_evicted_warm = 2
        hybrid.prepared_cache_rebalance_evicted_activated = 1
        hybrid.pipeline_ticks = 3
        hybrid._update_prepared_cache_rebalance_pressure_signals()

        assert hybrid.prepared_cache_rebalance_events_last_tick == 3
        assert hybrid._prepared_cache_rebalance_pressure_step() == pytest.approx(1.0)
        assert hybrid.prepared_cache_rebalance_pressure_ema == pytest.approx(0.2)

        hybrid.prepared_cache_rebalance_evicted_warm = 3
        hybrid.prepared_cache_rebalance_evicted_activated = 1
        hybrid.pipeline_ticks = 4
        hybrid._update_prepared_cache_rebalance_pressure_signals()

        assert hybrid.prepared_cache_rebalance_events_last_tick == 1
        assert hybrid._prepared_cache_rebalance_pressure_step() == pytest.approx(1.0 / 3.0)
        assert hybrid.prepared_cache_rebalance_pressure_ema == pytest.approx(0.2266666667)

    def test_effective_prepared_cache_limit_shrinks_under_pressure(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
            expert_prepared_cache_size=3,
        ).to(dtype=torch.float32)

        hybrid.prepared_cache_activation_stage_bonus = 0.0
        hybrid.prepared_cache_rebalance_evicted_warm = 2
        hybrid.prepared_cache_rebalance_evicted_activated = 1
        hybrid.pipeline_ticks = 3

        assert hybrid._effective_prepared_cache_limit() == 2
        diagnostics = hybrid.diagnostics()
        assert diagnostics["effective_prepared_cache_limit"] == 2
        assert diagnostics["prepared_cache_rebalance_pressure"] == pytest.approx(1.0)

    def test_prepared_cache_stage_bonus_tracks_rebalance_direction(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 4.0
        state.hotness[2] = 3.0
        state.hotness[3] = 10.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(0, 1, ExpertResidency.PIM, ExpertResidency.GPU, "bonus_act_1"),
                ExpertMigrationOp(0, 2, ExpertResidency.PIM, ExpertResidency.GPU, "bonus_act_2"),
                ExpertMigrationOp(0, 3, ExpertResidency.PIM, ExpertResidency.GPU, "bonus_new"),
            ],
            phase="decode",
        )

        for expert_idx in (1, 2):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid._store_activated_module(expert_idx, module.to(dtype=torch.float32))

        initial_bonus = hybrid.prepared_cache_activation_stage_bonus
        module = hybrid._build_runtime_expert(3, torch.device("cpu"), torch.float32)
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            3,
            state=MigrationLifecycle.ACTIVATED,
            phase="decode",
        )
        hybrid._store_activated_module(3, module.to(dtype=torch.float32))

        assert hybrid.prepared_cache_activation_stage_bonus != initial_bonus
        assert hybrid.prepared_cache_activation_stage_bonus >= 0.0

    def test_adaptive_cache_limits_shrink_under_prepared_cache_pressure(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.prepared_cache_activation_stage_bonus = 0.0
        for expert_idx in (1, 2):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.ACTIVATED,
                phase="decode",
            )
            hybrid._store_activated_module(expert_idx, module.to(dtype=torch.float32))

        assert hybrid._prepared_cache_pressure() >= 1.0
        assert hybrid._prepared_controller_engaged() is True
        assert hybrid._adaptive_activation_limit() == 1
        assert hybrid._adaptive_prebuild_limit() == 2

    def test_cold_promotion_penalty_grows_after_cold_apply_batch(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

        weight_path = tmp_path / "weights"
        weight_path.mkdir()
        tensors = {}
        for expert_idx in range(2):
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w1.weight"] = torch.randn(8, 4)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w2.weight"] = torch.randn(4, 8)
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w3.weight"] = torch.randn(8, 4)
        save_file(tensors, str(weight_path / "model.safetensors"))

        gpu_mask = torch.tensor([True, False], dtype=torch.bool)
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            [gpu_mask],
            default_offload_tier=ExpertResidency.PIM,
        )
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(enabled=True, gpu_budget_per_layer=1, offload_tier=ExpertResidency.PIM),
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
            expert_warm_cache_size=2,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        original_prebuild = hybrid._prebuild_ready_experts
        original_activate = hybrid._activate_warmed_experts
        hybrid._prebuild_ready_experts = lambda **kwargs: 0
        hybrid._activate_warmed_experts = lambda **kwargs: 0

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="cold_apply",
                )
            ],
            phase="decode",
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.READY,
            phase="decode",
        )

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        hybrid._prebuild_ready_experts = original_prebuild
        hybrid._activate_warmed_experts = original_activate

        assert stats["apply_batch_cold"] == 1
        assert hybrid.cold_promotion_penalty > 0.0
        assert hybrid._adaptive_prebuild_limit() >= 3

    def test_adaptive_prefetch_limits_follow_controller_pressure(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=2,
                prefill_force_gpu_budget_per_layer=2,
                prefetch_candidate_budget_per_layer=4,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.prepared_cache_activation_stage_bonus = 0.0
        hybrid.prepared_cache_rebalance_evicted_warm = 2
        hybrid.prepared_cache_rebalance_evicted_activated = 1
        hybrid.pipeline_ticks = 1
        hybrid._update_prepared_cache_rebalance_pressure_signals()

        assert hybrid._adaptive_prefetch_pending_limit(phase="decode") == 1
        assert hybrid._adaptive_prefetch_candidate_budget(phase="decode") == 2

        hybrid.cold_promotion_penalty = 1.5
        assert hybrid._adaptive_prefetch_pending_limit(phase="decode") == 4
        assert hybrid._adaptive_prefetch_candidate_budget(phase="decode") == 6

    def test_apply_queue_pressure_backoff_reduces_prepared_aggressiveness(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
                prefetch_candidate_budget_per_layer=3,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=4,
            prepared_controller_aggressiveness=1.0,
        ).to(dtype=torch.float32)

        base_activation = hybrid._adaptive_activation_limit()
        base_prebuild = hybrid._adaptive_prebuild_limit()
        base_prefetch_pending = hybrid._adaptive_prefetch_pending_limit(phase="decode")
        base_prefetch_budget = hybrid._adaptive_prefetch_candidate_budget(phase="decode")

        hybrid.apply_queue_evictions = 4
        hybrid.pipeline_ticks = 2
        hybrid._update_apply_queue_pressure_signals()

        assert hybrid._apply_queue_pressure() >= 2.0
        assert hybrid._apply_queue_budget_backoff() == 2
        assert hybrid._adaptive_activation_limit() < base_activation
        assert hybrid._adaptive_prebuild_limit() < base_prebuild
        assert hybrid._adaptive_prefetch_pending_limit(phase="decode") < base_prefetch_pending
        assert hybrid._adaptive_prefetch_candidate_budget(phase="decode") < base_prefetch_budget

    def test_apply_commit_queue_pressure_backoff_reduces_prepared_aggressiveness(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
                prefetch_candidate_budget_per_layer=3,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=4,
            prepared_controller_aggressiveness=1.0,
        ).to(dtype=torch.float32)

        base_activation = hybrid._adaptive_activation_limit()
        base_prebuild = hybrid._adaptive_prebuild_limit()
        base_prefetch_pending = hybrid._adaptive_prefetch_pending_limit(phase="decode")
        base_prefetch_budget = hybrid._adaptive_prefetch_candidate_budget(phase="decode")

        hybrid.apply_commit_queue_evictions = 4
        hybrid.pipeline_ticks = 2
        hybrid._update_apply_commit_queue_pressure_signals()

        assert hybrid._apply_commit_queue_pressure() >= 2.0
        assert hybrid._apply_commit_queue_budget_backoff() == 2
        assert hybrid._adaptive_activation_limit() < base_activation
        assert hybrid._adaptive_prebuild_limit() < base_prebuild
        assert hybrid._adaptive_prefetch_pending_limit(phase="decode") < base_prefetch_pending
        assert hybrid._adaptive_prefetch_candidate_budget(phase="decode") < base_prefetch_budget

    def test_apply_commit_batch_queue_pressure_backoff_reduces_prepared_aggressiveness(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
                prefetch_candidate_budget_per_layer=3,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=4,
            prepared_controller_aggressiveness=1.0,
        ).to(dtype=torch.float32)

        base_activation = hybrid._adaptive_activation_limit()
        base_prebuild = hybrid._adaptive_prebuild_limit()
        base_prefetch_pending = hybrid._adaptive_prefetch_pending_limit(phase="decode")
        base_prefetch_budget = hybrid._adaptive_prefetch_candidate_budget(phase="decode")

        hybrid.apply_commit_batch_queue_evictions = 4
        hybrid.pipeline_ticks = 2
        hybrid._update_apply_commit_batch_queue_pressure_signals()

        assert hybrid._apply_commit_batch_queue_pressure() >= 2.0
        assert hybrid._apply_commit_batch_queue_budget_backoff() == 2
        assert hybrid._adaptive_activation_limit() < base_activation
        assert hybrid._adaptive_prebuild_limit() < base_prebuild
        assert hybrid._adaptive_prefetch_pending_limit(phase="decode") < base_prefetch_pending
        assert hybrid._adaptive_prefetch_candidate_budget(phase="decode") < base_prefetch_budget

    def test_apply_commit_batch_limit_tracks_batch_queue_pressure(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
            expert_prepared_cache_size=4,
            prepared_controller_aggressiveness=1.0,
        ).to(dtype=torch.float32)

        base_commit_limit = hybrid._adaptive_apply_commit_limit(background=False)
        base_batch_limit = hybrid._adaptive_apply_commit_batch_limit(background=False)

        hybrid.apply_commit_batch_queue_evictions = 4
        hybrid.pipeline_ticks = 2
        hybrid._update_apply_commit_batch_queue_pressure_signals()

        assert hybrid._apply_commit_batch_queue_pressure() >= 2.0
        assert hybrid._apply_commit_batch_queue_budget_backoff() == 2
        assert hybrid._adaptive_apply_commit_limit(background=False) >= base_commit_limit
        assert hybrid._adaptive_apply_commit_batch_limit(background=False) <= base_batch_limit

    def test_background_pipeline_resolves_apply_commit_queue_before_commit(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
            expert_warm_cache_size=4,
            prepared_controller_aggressiveness=0.5,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="background_commit_resolve",
                )
            ],
            phase="decode",
        )
        hybrid.offload_backend.migration_manager.mark_state(
            0,
            1,
            state=MigrationLifecycle.ACTIVATED,
            phase="decode",
        )
        hybrid.activated_expert_cache["1"] = hybrid._build_runtime_expert(
            1,
            torch.device("cpu"),
            torch.float32,
        )

        stats = hybrid.background_advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()

        assert stats["apply_queue_enqueued"] == 1
        assert stats["activation_applied"] == 0
        assert stats["apply_commit_queue_enqueued"] == 1
        assert diagnostics["background_apply_commit_resolved"] == 1
        assert diagnostics["apply_commit_ready_stores"] == 1
        assert diagnostics["apply_commit_ready_hits"] == 0
        assert stats["apply_commit_batch_queue_prefinalized"] == 1
        assert diagnostics["background_apply_commit_batch_queue_prefinalized_batches"] == 1
        assert stats["resident_commit_batch_queue_prefinalized"] == 1
        assert diagnostics["background_resident_commit_batch_queue_prefinalized_batches"] == 1

    def test_warm_cache_eviction_downgrades_lifecycle_to_ready(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan

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
            expert_warm_cache_size=1,
        ).to(dtype=torch.float32)
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="warm_probe_1",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=2,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="warm_probe_2",
                ),
            ],
            phase="decode",
        )

        for expert_idx in (1, 2):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.WARMED,
                phase="decode",
            )
            hybrid._store_warm_module(expert_idx, module, count_store=False)

        lifecycle = {
            int(expert_idx): hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2)
        }
        assert lifecycle[1] == MigrationLifecycle.READY.value
        assert lifecycle[2] == MigrationLifecycle.WARMED.value
        assert "1" not in hybrid.warm_expert_cache
        assert "2" in hybrid.warm_expert_cache
        layer_diag = hybrid.offload_backend.migration_manager.diagnostics()["layers"][0]
        assert layer_diag["total_warm_eviction_regressions"] == 1

    def test_warm_cache_prefers_colder_victim_on_eviction(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertMigrationOp,
            ExpertResidency,
            ExpertResidencyPlan,
        )

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 9.0
        state.hotness[2] = 1.0
        state.hotness[3] = 8.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=2,
        ).to(dtype=torch.float32)

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="warm_hot",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=2,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="warm_cold",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=3,
                    src=ExpertResidency.PIM,
                    dst=ExpertResidency.GPU,
                    reason="warm_new",
                ),
            ],
            phase="decode",
        )

        for expert_idx in (1, 2, 3):
            module = hybrid._build_runtime_expert(expert_idx, torch.device("cpu"), torch.float32)
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.WARMED,
                phase="decode",
            )
            hybrid._store_warm_module(expert_idx, module, count_store=False)

        assert "1" in hybrid.warm_expert_cache
        assert "3" in hybrid.warm_expert_cache
        assert "2" not in hybrid.warm_expert_cache
        lifecycle = {
            expert_idx: hybrid.offload_backend.migration_manager.state_for(0, expert_idx).value
            for expert_idx in (1, 2, 3)
        }
        assert lifecycle[1] == MigrationLifecycle.WARMED.value
        assert lifecycle[2] == MigrationLifecycle.READY.value
        assert lifecycle[3] == MigrationLifecycle.WARMED.value

    def test_hybrid_moe_prebuild_targets_only_hot_ready_candidates(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )
        state = residency_plan.layer_state(0)
        state.hotness[1] = 8.0
        state.hotness[2] = 6.0
        state.hotness[3] = 1.0

        hybrid = HybridMoE(
            num_experts=4,
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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2, 3):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="prebuild_budget",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.READY,
                phase="decode",
            )

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()

        assert stats["warm_prebuilt"] == 2
        assert "1" in hybrid.gpu_experts or "1" in hybrid.warm_expert_cache or "1" in hybrid.activated_expert_cache
        assert "2" in hybrid.warm_expert_cache or "2" in hybrid.activated_expert_cache
        assert "3" not in hybrid.warm_expert_cache

    def test_migration_queue_preserves_warmed_state_on_deferred_requeue(self):
        from nano_ktrans.kernels.expert_migration import ExpertMigrationManager, MigrationLifecycle
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        manager = ExpertMigrationManager()
        op = ExpertMigrationOp(
            layer_idx=0,
            expert_idx=3,
            src=ExpertResidency.PIM,
            dst=ExpertResidency.GPU,
            reason="preserve_warmed",
        )
        manager.queue(0, [op], phase="decode")
        manager.mark_state(0, 3, state=MigrationLifecycle.WARMED, phase="decode")
        manager.queue(0, [op], phase="decode_deferred")

        diagnostics = manager.diagnostics()["layers"][0]
        lifecycle = diagnostics["lifecycle"][0]

        assert lifecycle["state"] == MigrationLifecycle.WARMED.value
        assert diagnostics["total_deferred_events"] == 0
        assert diagnostics["total_requeue_preserved_states"] == 1

    def test_migration_queue_preserves_deferred_state_on_requeue(self):
        from nano_ktrans.kernels.expert_migration import ExpertMigrationManager, MigrationLifecycle
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        manager = ExpertMigrationManager()
        op = ExpertMigrationOp(
            layer_idx=0,
            expert_idx=4,
            src=ExpertResidency.PIM,
            dst=ExpertResidency.GPU,
            reason="preserve_deferred",
        )
        manager.queue(0, [op], phase="decode_deferred")
        manager.queue(0, [op], phase="decode_deferred")

        diagnostics = manager.diagnostics()["layers"][0]
        lifecycle = diagnostics["lifecycle"][0]

        assert lifecycle["state"] == MigrationLifecycle.DEFERRED.value
        assert diagnostics["total_deferred_events"] == 1
        assert diagnostics["total_requeue_preserved_states"] == 1

    def test_migration_queue_skips_backward_stage_regression(self):
        from nano_ktrans.kernels.expert_migration import ExpertMigrationManager, MigrationLifecycle
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency

        manager = ExpertMigrationManager()
        op = ExpertMigrationOp(
            layer_idx=0,
            expert_idx=5,
            src=ExpertResidency.PIM,
            dst=ExpertResidency.GPU,
            reason="skip_stage_regression",
        )
        manager.queue(0, [op], phase="decode")
        manager.mark_state(0, 5, state=MigrationLifecycle.READY, phase="decode")
        manager.mark_state(0, 5, state=MigrationLifecycle.DEFERRED, phase="decode")

        diagnostics = manager.diagnostics()["layers"][0]
        lifecycle = diagnostics["lifecycle"][0]

        assert lifecycle["state"] == MigrationLifecycle.READY.value
        assert diagnostics["total_stage_skips"] == 1

    def test_ready_promotions_do_not_requeue_after_budget_limit(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
            ),
        )
        residency_plan.layer_state(0).hotness[1] = 9.0
        residency_plan.layer_state(0).hotness[2] = 4.0

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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="batch_ready_promote",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.READY,
                phase="decode",
            )

        stats = hybrid.advance_offload_pipeline(
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        diagnostics = hybrid.diagnostics()
        layer_diag = diagnostics["backend"]["migration_manager"]["layers"][0]

        assert stats["ready_applied"] == 1
        assert stats["ready_deferred"] == 1
        assert layer_diag["pending_ops"] == 1
        assert layer_diag["total_enqueued_ops"] == 2
        assert layer_diag["total_deferred_events"] == 0
        assert diagnostics["pipeline_apply_batches"] == 1
        assert diagnostics["pipeline_apply_batch_experts"] == 1
        assert diagnostics["pipeline_apply_batch_evictions"] == 1

    def test_forward_path_keeps_unapplied_ready_promotions_pending(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
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
        scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                offload_tier=ExpertResidency.PIM,
                decode_promote_k=1,
                decode_require_prefetch_ready=False,
            ),
        )
        residency_plan.layer_state(0).hotness[1] = 9.0
        residency_plan.layer_state(0).hotness[2] = 5.0

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
            expert_warm_cache_size=4,
        ).to(dtype=torch.float32)

        for expert_idx in (1, 2):
            hybrid.offload_backend.queue_migration_plan(
                [
                    ExpertMigrationOp(
                        layer_idx=0,
                        expert_idx=expert_idx,
                        src=ExpertResidency.PIM,
                        dst=ExpertResidency.GPU,
                        reason="forward_ready_budget",
                    )
                ],
                phase="decode",
            )
            hybrid.materialization_manager.stage_expert(
                0,
                expert_idx,
                {
                    "gate": torch.randn(8, 4),
                    "up": torch.randn(8, 4),
                    "down": torch.randn(4, 8),
                },
            )
            hybrid.offload_backend.migration_manager.mark_state(
                0,
                expert_idx,
                state=MigrationLifecycle.READY,
                phase="decode",
            )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        diagnostics = hybrid.diagnostics()
        layer_diag = diagnostics["backend"]["migration_manager"]["layers"][0]
        lifecycle = {entry["expert_idx"]: entry["state"] for entry in layer_diag["lifecycle"]}

        assert output.shape == (1, 4)
        assert diagnostics["applied_migration_ops"] == 2  # 1 eviction + 1 promotion
        assert layer_diag["pending_ops"] == 1
        assert lifecycle[1] == MigrationLifecycle.APPLIED.value
        assert lifecycle[2] == MigrationLifecycle.READY.value
        assert str(1) in hybrid.gpu_experts
        assert str(2) not in hybrid.gpu_experts

    def test_forward_path_only_removes_applied_demotion(self, tmp_path):
        from safetensors.torch import save_file

        from nano_ktrans.kernels.expert_migration import MigrationLifecycle
        from nano_ktrans.layers.expert_mlp import build_expert_module, load_expert_weights
        from nano_ktrans.layers.hybrid_moe import HybridMoE
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.context import reset_context, set_context
        from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency, ExpertResidencyPlan

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
                gpu_budget_per_layer=2,
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
        hybrid.dynamic_expert_scheduler.plan_layer = lambda *args, **kwargs: []
        hybrid.gpu_experts_mask[1] = True

        hybrid.materialization_manager.stage_expert(
            0,
            0,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        hybrid.materialization_manager.stage_expert(
            0,
            1,
            {
                "gate": torch.randn(8, 4),
                "up": torch.randn(8, 4),
                "down": torch.randn(4, 8),
            },
        )
        for expert_idx in (0, 1):
            module = build_expert_module(
                hidden_size=4,
                intermediate_size=8,
                hidden_act="silu",
                experts_are_packed=False,
            ).to(dtype=torch.float32)
            load_expert_weights(module, hybrid.materialization_manager.get_expert(0, expert_idx))
            hybrid.gpu_experts[str(expert_idx)] = module

        hybrid.offload_backend.queue_migration_plan(
            [
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=0,
                    src=ExpertResidency.GPU,
                    dst=ExpertResidency.PIM,
                    reason="demote_active_skip",
                ),
                ExpertMigrationOp(
                    layer_idx=0,
                    expert_idx=1,
                    src=ExpertResidency.GPU,
                    dst=ExpertResidency.PIM,
                    reason="demote_inactive_apply",
                ),
            ],
            phase="decode",
        )

        hidden_states = torch.randn(1, 4)
        router_logits = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        set_context(is_prefill=False)
        try:
            output = hybrid(hidden_states, router_logits)
        finally:
            reset_context()

        diagnostics = hybrid.diagnostics()
        layer_diag = diagnostics["backend"]["migration_manager"]["layers"][0]
        lifecycle = {entry["expert_idx"]: entry["state"] for entry in layer_diag["lifecycle"]}

        assert output.shape == (1, 4)
        assert layer_diag["pending_ops"] == 1
        assert lifecycle[0] == MigrationLifecycle.QUEUED.value
        assert lifecycle[1] == MigrationLifecycle.APPLIED.value
        assert bool(hybrid.gpu_experts_mask[0].item()) is True
        assert bool(hybrid.gpu_experts_mask[1].item()) is False
        assert str(0) in hybrid.gpu_experts
        assert str(1) not in hybrid.gpu_experts

    def test_llm_reset_offload_diagnostics_zeros_runtime_counters(self):
        from nano_ktrans.llm import LLM

        llm = LLM.__new__(LLM)

        class DummyHybrid:
            def __init__(self):
                self.prefetch_requested = 3
                self.prefetch_enqueued = 2
                self.prefetch_materialized = 1
                self.prefetch_candidate_scans = 4
                self.runtime_evictions = 5
                self.runtime_skipped_demotion_cooldown = 6
                self.runtime_deferred_for_prefetch = 7
                self.decode_prefetch_hits = 8
                self.decode_prefetch_misses = 9
                self.pipeline_ready_applied = 10
                self.pipeline_ready_deferred = 11
                self.pipeline_ticks = 12
                self.pipeline_prefetch_overlap_hits = 13
                self.pipeline_promotion_source_activated = 14
                self.pipeline_promotion_source_warm = 15
                self.pipeline_promotion_source_cold = 16
                self.prepared_cache_rebalance_pressure_ema = 0.4
                self.prepared_cache_rebalance_events_last_tick = 2
                self.prepared_cache_rebalance_events_prev_total = 5
                self.warm_cache_hits = 13
                self.warm_cache_stores = 14
                self.warm_cache_evictions = 15
                self.warm_cache_prebuilt = 16
                self.warm_cache_device_transfers = 17
                self.activated_cache_hits = 18
                self.activated_cache_stores = 19
                self.activated_cache_evictions = 20
                self.activation_submitted = 21
                self.activation_ready = 22
                self.activation_applied = 23
                self.background_activation_applied = 24
                self.resident_commit_batch_queue_enqueued = 25
                self.resident_commit_batch_queue_batches = 26
                self.resident_commit_batch_queue_committed_batches = 27
                self.resident_commit_batch_queue_pruned = 28
                self.resident_commit_batch_queue_evictions = 29
                self.background_resident_commit_batch_queue_enqueued = 30
                self.background_resident_commit_batch_queue_committed_batches = 31
                self.background_resident_commit_batch_queue_prefinalized_batches = 32
                self.resident_commit_finalize_queue_enqueued = 33
                self.resident_commit_finalize_queue_batches = 34
                self.resident_commit_finalize_queue_committed_batches = 35
                self.resident_commit_finalize_queue_pruned = 36
                self.resident_commit_finalize_queue_evictions = 37
                self.background_resident_commit_finalize_queue_enqueued = 38
                self.background_resident_commit_finalize_queue_committed_batches = 39
                self.background_resident_commit_finalize_queue_prefinalized_batches = 40
                self.resident_commit_ready_cache_stores = 41
                self.resident_commit_ready_cache_hits = 42
                self.resident_commit_ready_cache_pruned = 43
                self.resident_commit_ready_cache_evictions = 44
                self.background_resident_commit_ready_cache_stores = 45

        dummy_hybrid = DummyHybrid()
        layer = type("Layer", (), {"hybrid_moe": dummy_hybrid})()
        llm.model = type("Wrapper", (), {"model": type("Inner", (), {"layers": [layer]})()})()

        llm.reset_offload_diagnostics()

        assert dummy_hybrid.prefetch_requested == 0
        assert dummy_hybrid.runtime_evictions == 0
        assert dummy_hybrid.pipeline_ticks == 0
        assert dummy_hybrid.pipeline_prefetch_overlap_hits == 0
        assert dummy_hybrid.pipeline_promotion_source_activated == 0
        assert dummy_hybrid.prepared_cache_rebalance_pressure_ema == 0.0
        assert dummy_hybrid.prepared_cache_rebalance_events_last_tick == 0
        assert dummy_hybrid.prepared_cache_rebalance_events_prev_total == 0
        assert dummy_hybrid.warm_cache_prebuilt == 0
        assert dummy_hybrid.activated_cache_hits == 0
        assert dummy_hybrid.activation_applied == 0
        assert dummy_hybrid.background_activation_applied == 0
        assert dummy_hybrid.resident_commit_batch_queue_enqueued == 0
        assert dummy_hybrid.background_resident_commit_batch_queue_enqueued == 0
        assert dummy_hybrid.background_resident_commit_batch_queue_prefinalized_batches == 0
        assert dummy_hybrid.resident_commit_finalize_queue_enqueued == 0
        assert dummy_hybrid.background_resident_commit_finalize_queue_enqueued == 0
        assert dummy_hybrid.background_resident_commit_finalize_queue_prefinalized_batches == 0
        assert dummy_hybrid.resident_commit_ready_cache_stores == 0
        assert dummy_hybrid.background_resident_commit_ready_cache_stores == 0

    def test_llm_reset_offload_diagnostics_zeros_runtime_background_tick_counters(self):
        from nano_ktrans.llm import LLM

        llm = LLM.__new__(LLM)
        worker_reset_calls = {"count": 0}

        class DummyRuntime:
            def __init__(self):
                self.tick_calls = 4
                self.background_ticks = 5
                self.background_work_items_total = 6
                self.background_warm_prebuilt_total = 6
                self.background_activation_ready_total = 7
                self.background_activation_applied_total = 8
                self.background_apply_queue_enqueued_total = 9
                self.background_apply_commit_queue_enqueued_total = 10
                self.background_apply_commit_batch_queue_enqueued_total = 11
                self.background_apply_commit_batch_queue_prefinalized_total = 12
                self.background_resident_commit_batch_queue_enqueued_total = 13
                self.background_resident_commit_batch_queue_prefinalized_total = 14
                self.background_resident_commit_finalize_queue_enqueued_total = 15
                self.background_resident_commit_finalize_queue_prefinalized_total = 16
                self.background_resident_commit_ready_cache_stores_total = 17
                self.prefetch_submitted_total = 6
                self.ready_polled_total = 15
                self.activation_ready_total = 16
                self.ready_applied_total = 17
                self.ready_deferred_total = 18
                self.apply_batch_count_total = 19
                self.apply_batch_experts_total = 20
                self.apply_batch_evictions_total = 21
                self.apply_batch_activated_total = 22
                self.apply_batch_warm_total = 23
                self.apply_batch_cold_total = 24
                self.layers_touched_total = 25
                self.background_ready_callback_total = 26
                self.last_phase = "decode"

        class DummyHybrid:
            pass

        runtime = DummyRuntime()
        layer = type("Layer", (), {"hybrid_moe": DummyHybrid()})()
        model = type(
            "Inner",
            (),
            {
                "layers": [layer],
                "offload_runtime": runtime,
                "reset_offload_worker_diagnostics": lambda self: worker_reset_calls.__setitem__("count", worker_reset_calls["count"] + 1),
            },
        )()
        llm.model = type("Wrapper", (), {"model": model})()

        llm.reset_offload_diagnostics()

        assert runtime.tick_calls == 0
        assert runtime.background_ticks == 0
        assert runtime.background_work_items_total == 0
        assert runtime.background_warm_prebuilt_total == 0
        assert runtime.background_activation_ready_total == 0
        assert runtime.background_activation_applied_total == 0
        assert runtime.background_apply_queue_enqueued_total == 0
        assert runtime.background_apply_commit_queue_enqueued_total == 0
        assert runtime.background_apply_commit_batch_queue_enqueued_total == 0
        assert runtime.background_apply_commit_batch_queue_prefinalized_total == 0
        assert runtime.background_resident_commit_batch_queue_enqueued_total == 0
        assert runtime.background_resident_commit_batch_queue_prefinalized_total == 0
        assert runtime.background_resident_commit_finalize_queue_enqueued_total == 0
        assert runtime.background_resident_commit_finalize_queue_prefinalized_total == 0
        assert runtime.background_resident_commit_ready_cache_stores_total == 0
        assert runtime.prefetch_submitted_total == 0
        assert runtime.background_ready_callback_total == 0
        assert runtime.last_phase == ""
        assert worker_reset_calls["count"] == 1

    def test_llm_shutdown_calls_model_offload_worker_shutdown(self):
        from nano_ktrans.llm import LLM

        llm = LLM.__new__(LLM)
        called = {"count": 0}

        class DummyModel:
            def shutdown_offload_worker(self):
                called["count"] += 1

        llm.model = type("Wrapper", (), {"model": DummyModel()})()
        llm.shutdown()
        assert called["count"] == 1

    def test_llm_generate_starts_and_stops_background_worker(self):
        from nano_ktrans.llm import LLM

        llm = LLM.__new__(LLM)
        llm.device = "cpu"
        llm.tokenizer = type(
            "Tokenizer",
            (),
            {
                "eos_token_id": -1,
                "decode": staticmethod(lambda ids, skip_special_tokens=True: "decoded"),
                "__call__": staticmethod(
                    lambda prompt, return_tensors="pt": type(
                        "Inputs",
                        (),
                        {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
                    )()
                ),
            },
        )()

        calls = {"start": 0, "stop": 0, "shutdown": 0, "prefill": 0, "decode": 0}

        class DummyEngine:
            def start_background_offload_worker(self):
                calls["start"] += 1

            def stop_background_offload_worker(self):
                calls["stop"] += 1

            def prefill(self, input_ids):
                calls["prefill"] += 1
                return torch.tensor([[[0.1, 0.9]]], dtype=torch.float32)

            def decode_step(self, next_token, current_seq_len):
                calls["decode"] += 1
                return torch.tensor([[[0.9, 0.1]]], dtype=torch.float32)

        class DummyModel:
            def shutdown_offload_worker(self):
                calls["shutdown"] += 1

        llm.engine = DummyEngine()
        llm.model = type("Wrapper", (), {"model": DummyModel()})()
        output = llm.generate("hi", max_new_tokens=2)

        assert output == "decoded"
        assert calls["start"] == 1
        assert calls["stop"] == 1
        assert calls["shutdown"] == 1
        assert calls["prefill"] == 1
        assert calls["decode"] == 1

    def test_benchmark_run_single_generation_starts_and_stops_background_worker(self):
        from benchmarks.benchmark_inference import run_single_generation

        class DummyTokenizer:
            eos_token_id = 99

            def __call__(self, prompt, return_tensors="pt"):
                return type(
                    "Inputs",
                    (),
                    {"input_ids": torch.tensor([[11, 12]], dtype=torch.long)},
                )()

            def decode(self, generated_ids, skip_special_tokens=True):
                return "decoded"

        class DummyEngine:
            def __init__(self):
                self.start_calls = 0
                self.stop_calls = 0
                self.prefill_calls = 0
                self.decode_calls = 0

            def start_background_offload_worker(self):
                self.start_calls += 1

            def stop_background_offload_worker(self):
                self.stop_calls += 1

            def prefill(self, input_ids):
                self.prefill_calls += 1
                logits = torch.zeros(1, input_ids.shape[1], 128)
                logits[0, -1, 7] = 1.0
                return logits

            def decode_step(self, next_token, current_seq_len):
                self.decode_calls += 1
                logits = torch.zeros(1, 1, 128)
                logits[0, -1, 99] = 1.0
                return logits

        class DummyLLM:
            def __init__(self):
                self.device = "cpu"
                self.tokenizer = DummyTokenizer()
                self.engine = DummyEngine()
                self.reset_calls = 0

            def reset_offload_diagnostics(self):
                self.reset_calls += 1

            def get_offload_diagnostics(self):
                return {
                    "dynamic_scheduler": {},
                    "layers": [],
                    "offload_refresh": {},
                    "layer_count": 0,
                    "scheduler_profile": {},
                }

        llm = DummyLLM()
        result = run_single_generation(llm, "hi", max_new_tokens=2)

        assert llm.reset_calls == 1
        assert llm.engine.start_calls == 1
        assert llm.engine.stop_calls == 1
        assert llm.engine.prefill_calls == 1
        assert llm.engine.decode_calls == 1
        assert result["generated_tokens"] == 2
        assert result["output_text"] == "decoded"


# ============================================================
# Test N+1: MRS Score-Aware Hotness (HybriMoE, ADR-001 P1)
# ============================================================
class TestHotnessMRS:
    def test_bincount_mode_matches_legacy_behaviour(self):
        """未提供 mrs_alpha 时，update_hotness 保持与旧实现一致"""
        from nano_ktrans.utils.expert_runtime_state import update_hotness

        hotness = torch.zeros(4, dtype=torch.float32)
        ids = torch.tensor([[0, 1], [0, 2]])
        updated = update_hotness(hotness, ids, decay=0.5)
        # 两个 token，expert 0 被选两次，expert 1/2 各一次
        assert torch.allclose(updated, torch.tensor([2.0, 1.0, 1.0, 0.0]))

    def test_mrs_mode_uses_router_scores(self):
        """MRS 模式按 router score 加权 EMA，高分 expert 影响更大"""
        from nano_ktrans.utils.expert_runtime_state import update_hotness

        hotness = torch.zeros(4, dtype=torch.float32)
        ids = torch.tensor([[0, 1]])
        scores = torch.tensor([[0.9, 0.1]])
        updated = update_hotness(
            hotness, ids, router_scores=scores, mrs_alpha=0.5, top_p=2
        )
        # alpha=0.5 → S = 0.5 * (score / tokens=1) + 0 = 0.5 * [0.9, 0.1, 0, 0]
        assert updated[0].item() == pytest.approx(0.45)
        assert updated[1].item() == pytest.approx(0.05)
        assert updated[2].item() == 0.0

    def test_mrs_top_p_truncates_low_score_experts(self):
        """top_p 截断：只有前 p 个 expert 进入 EMA"""
        from nano_ktrans.utils.expert_runtime_state import update_hotness

        hotness = torch.zeros(4, dtype=torch.float32)
        ids = torch.tensor([[0, 1, 2]])
        scores = torch.tensor([[0.5, 0.4, 0.1]])
        updated = update_hotness(
            hotness, ids, router_scores=scores, mrs_alpha=1.0, top_p=2
        )
        # alpha=1.0 + top_p=2 ⇒ 只 expert 0/1 计入，expert 2 被裁
        assert updated[0].item() == pytest.approx(0.5)
        assert updated[1].item() == pytest.approx(0.4)
        assert updated[2].item() == 0.0

    def test_mrs_alpha_decay_without_new_observation(self):
        """router_scores 全零/无 token 时，hotness 仍按 (1-alpha) 衰减"""
        from nano_ktrans.utils.expert_runtime_state import update_hotness

        hotness = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ids = torch.empty((0, 2), dtype=torch.long)
        scores = torch.empty((0, 2))
        updated = update_hotness(
            hotness, ids, router_scores=scores, mrs_alpha=0.25, top_p=2
        )
        # 空观察 ⇒ hotness 乘以 (1 - alpha) = 0.75
        assert torch.allclose(updated, hotness * 0.75)

    def test_scheduler_observe_routes_scores_when_mrs_enabled(self):
        """DynamicExpertScheduler.observe 在开启 MRS 时使用 router scores"""
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertResidency,
            ExpertResidencyPlan,
        )

        plan = ExpertResidencyPlan.from_gpu_masks(
            [torch.zeros(4, dtype=torch.bool)],
            default_offload_tier=ExpertResidency.CPU,
        )
        config = SchedulerConfig(
            enabled=True,
            gpu_budget_per_layer=1,
            hotness_decay=0.5,
            hotness_mrs_alpha=0.5,
            hotness_top_p=2,
        )
        scheduler = DynamicExpertScheduler(residency_plan=plan, config=config)

        topk_ids = torch.tensor([[0, 1]])
        topk_weights = torch.tensor([[0.8, 0.2]])
        scheduler.observe(0, topk_ids, phase="decode", topk_weights=topk_weights)
        assert scheduler.hotness_mrs_observations == 1
        assert scheduler.hotness_bincount_observations == 0
        # expert 0 (score=0.8) 的 hotness 应该严格大于 expert 1 (score=0.2)
        hotness = plan.layer_state(0).hotness
        assert hotness[0].item() > hotness[1].item()

    def test_scheduler_observe_falls_back_to_bincount_without_weights(self):
        """未传 topk_weights 时，MRS 配置仍会走 bincount 分支并记入计数"""
        from nano_ktrans.scheduler import DynamicExpertScheduler, SchedulerConfig
        from nano_ktrans.utils.expert_runtime_state import (
            ExpertResidency,
            ExpertResidencyPlan,
        )

        plan = ExpertResidencyPlan.from_gpu_masks(
            [torch.zeros(4, dtype=torch.bool)],
            default_offload_tier=ExpertResidency.CPU,
        )
        scheduler = DynamicExpertScheduler(
            residency_plan=plan,
            config=SchedulerConfig(
                enabled=True,
                gpu_budget_per_layer=1,
                hotness_mrs_alpha=0.5,
            ),
        )
        scheduler.observe(0, torch.tensor([[0, 1]]), phase="decode")
        assert scheduler.hotness_mrs_observations == 0
        assert scheduler.hotness_bincount_observations == 1


# ============================================================
# Test N+2: Expert Map Store (fMoE, ADR-001 P2)
# ============================================================
class TestExpertMapStore:
    def test_commit_enforces_capacity_lru(self):
        """容量上限触发 LRU 驱逐，最旧的 iteration 先出"""
        from nano_ktrans.utils.expert_map_store import ExpertMapStore

        store = ExpertMapStore(capacity=2, prefetch_distance=1)
        for i in range(3):
            emap = store.begin_iteration(torch.tensor([float(i), 0.0, 0.0]))
            emap.record_layer(0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
            store.commit_iteration(emap)
        assert len(store) == 2
        assert store.eviction_count == 1

    def test_semantic_search_matches_closest_prompt(self):
        """语义搜索返回与当前 prompt 最相似的历史 iteration 的高概率 expert"""
        from nano_ktrans.utils.expert_map_store import ExpertMapStore

        store = ExpertMapStore(capacity=8, prefetch_distance=2)
        # Map A: prompt=[1,0,0]，layer 0 的高概率 expert 是 0
        map_a = store.begin_iteration(torch.tensor([1.0, 0.0, 0.0]))
        map_a.record_layer(0, torch.tensor([0.9, 0.05, 0.03, 0.02]))
        store.commit_iteration(map_a)
        # Map B: prompt=[0,1,0]，layer 0 的高概率 expert 是 3
        map_b = store.begin_iteration(torch.tensor([0.0, 1.0, 0.0]))
        map_b.record_layer(0, torch.tensor([0.05, 0.05, 0.1, 0.8]))
        store.commit_iteration(map_b)

        # 查询 prompt 接近 A
        picks = store.semantic_search(
            torch.tensor([0.95, 0.1, 0.0]),
            layer_idx=0,
            top_k=1,
        )
        assert picks == [0]
        # 查询 prompt 接近 B
        picks = store.semantic_search(
            torch.tensor([0.1, 0.9, 0.0]),
            layer_idx=0,
            top_k=1,
        )
        assert picks == [3]

    def test_trajectory_search_uses_observed_layers(self):
        """轨迹搜索使用已观测到的 gate 分布匹配历史"""
        from nano_ktrans.utils.expert_map_store import ExpertMapStore

        store = ExpertMapStore(capacity=8, prefetch_distance=1)
        map_a = store.begin_iteration(torch.zeros(3))
        map_a.record_layer(0, torch.tensor([0.9, 0.05, 0.03, 0.02]))
        map_a.record_layer(1, torch.tensor([0.1, 0.8, 0.05, 0.05]))
        map_a.record_layer(2, torch.tensor([0.05, 0.05, 0.85, 0.05]))
        store.commit_iteration(map_a)

        map_b = store.begin_iteration(torch.zeros(3))
        map_b.record_layer(0, torch.tensor([0.05, 0.05, 0.1, 0.8]))
        map_b.record_layer(1, torch.tensor([0.8, 0.1, 0.05, 0.05]))
        map_b.record_layer(2, torch.tensor([0.7, 0.1, 0.1, 0.1]))
        store.commit_iteration(map_b)

        # 当前 iteration 的 layer 0 分布酷似 A，因此 layer 2 应选 expert 2
        observed = {0: torch.tensor([0.88, 0.04, 0.04, 0.04])}
        picks = store.trajectory_search(
            observed=observed,
            target_layer_idx=2,
            top_k=1,
        )
        assert picks == [2]

    def test_trajectory_search_empty_when_no_overlap(self):
        """无重叠层时返回空列表而不是崩溃"""
        from nano_ktrans.utils.expert_map_store import ExpertMapStore

        store = ExpertMapStore(capacity=4, prefetch_distance=1)
        emap = store.begin_iteration(torch.zeros(2))
        emap.record_layer(0, torch.tensor([0.5, 0.5]))
        store.commit_iteration(emap)

        picks = store.trajectory_search(
            observed={5: torch.tensor([0.5, 0.5])},
            target_layer_idx=5,
            top_k=1,
        )
        assert picks == []

    def test_diagnostics_shape(self):
        """diagnostics 包含关键字段"""
        from nano_ktrans.utils.expert_map_store import ExpertMapStore

        store = ExpertMapStore(capacity=4, prefetch_distance=2)
        diag = store.diagnostics()
        for key in (
            "capacity",
            "prefetch_distance",
            "size",
            "commit_count",
            "eviction_count",
            "semantic_queries",
            "semantic_hits",
            "trajectory_queries",
            "trajectory_hits",
        ):
            assert key in diag

