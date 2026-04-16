import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from nano_ktrans.models.mixtral import MixtralForCausalLM
from nano_ktrans.models.config import GenericMoeConfig, adapt_config_to_checkpoint
from nano_ktrans.scheduler import (
    DynamicExpertScheduler,
    SCHEDULER_PROFILE_BASELINE,
    SchedulerConfig,
    apply_scheduler_overrides,
    resolve_prepared_controller_aggressiveness,
    resolve_scheduler_profile,
    resolve_prepared_cache_budget,
    scheduler_profile_summary,
)
from nano_ktrans.utils.loader import load_model
from nano_ktrans.utils.expert_selection import (
    generate_gpu_experts_masks,
    uniform_gpu_experts_masks,
)
from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan
from nano_ktrans.engine.simple_engine import SimpleEngine

class LLM:
    """
    User-facing class for interacting with the nano-ktrans Mixtral model.

    专家放置策略：
    - 提供 activation_freq: 基于激活频率的数据驱动选择（推荐）
    - 不提供:              均匀选择前 N 个专家（fallback）
    """
    def __init__(
        self, 
        model_path: str, 
        max_seq_len: int = 2048, 
        device: Optional[str] = None,
        num_gpu_experts: Optional[int] = None,
        chunk_size: int = 512,
        offload_backend: str = "cpu",
        offload_backend_kwargs: Optional[dict] = None,
        activation_freq: Optional[torch.Tensor] = None,
        enable_dynamic_expert_scheduler: bool = False,
        scheduler_gpu_budget_per_layer: Optional[int] = None,
        scheduler_hotness_decay: float = 0.95,
        scheduler_offload_tier: str = "pim",
        scheduler_prefill_force_gpu_budget_per_layer: Optional[int] = None,
        scheduler_prefill_offload_threshold_tokens: int = 8,
        scheduler_decode_promote_k: int = 2,
        scheduler_prefill_collect_only: Optional[bool] = None,
        scheduler_step_stride_prefill: Optional[int] = None,
        scheduler_step_stride_decode: Optional[int] = None,
        scheduler_demotion_idle_steps: Optional[int] = None,
        scheduler_migration_cooldown_steps: Optional[int] = None,
        scheduler_decode_require_prefetch_ready: Optional[bool] = None,
        scheduler_prefetch_candidate_budget_per_layer: Optional[int] = None,
        scheduler_prepared_cache_budget_per_layer: Optional[int] = None,
        scheduler_profile: str = SCHEDULER_PROFILE_BASELINE,
        enable_background_offload_worker: bool = False,
        background_offload_poll_interval_seconds: float = 0.005,
    ):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no CUDA runtime is available.")

        # 自动解析 HF repo_id 为本地路径
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                model_path,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.txt",
                    "*.model",
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                ],
            )

        self.model_path = model_path
        self.device = device
        self.max_seq_len = max_seq_len
        self.offload_backend = offload_backend
        self.offload_backend_kwargs = offload_backend_kwargs or {}
        
        # print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # print(f"Loading config from {model_path}...")
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        
        config = GenericMoeConfig.from_hf_config(hf_config)
        config = adapt_config_to_checkpoint(config, model_path)
        if config.attention_backend != "standard":
            raise NotImplementedError(
                f"Model type '{getattr(hf_config, 'model_type', 'unknown')}' uses an attention backend "
                f"that nano-ktrans does not implement yet. DeepSeek-V2/V3 style MLA still needs a "
                f"separate adaptation pass before expert offload can be tested."
        )
        
        # ===== GPU 专家选择策略 =====
        effective_num_gpu_experts = config.num_local_experts if num_gpu_experts is None else num_gpu_experts
        effective_num_gpu_experts = max(0, min(effective_num_gpu_experts, config.num_local_experts))

        if config.num_local_experts > 0 and effective_num_gpu_experts == config.num_local_experts:
            layer_gpu_expert_masks = uniform_gpu_experts_masks(
                config.num_hidden_layers, config.num_local_experts, config.num_local_experts
            )
        elif config.num_local_experts > 0 and activation_freq is not None:
            # 数据驱动：基于激活频率选择热门专家（ktransformers 核心策略）
            layer_gpu_expert_masks = generate_gpu_experts_masks(activation_freq, effective_num_gpu_experts)
        elif config.num_local_experts > 0:
            # Fallback：均匀选择前 N 个专家
            layer_gpu_expert_masks = uniform_gpu_experts_masks(
                config.num_hidden_layers, config.num_local_experts, effective_num_gpu_experts
            )
        else:
            layer_gpu_expert_masks = [
                torch.zeros(0, dtype=torch.bool) for _ in range(config.num_hidden_layers)
            ]

        default_offload_tier = ExpertResidency.PIM if scheduler_offload_tier.lower() == "pim" else ExpertResidency.CPU
        residency_plan = ExpertResidencyPlan.from_gpu_masks(
            layer_gpu_expert_masks,
            default_offload_tier=default_offload_tier,
        )
        gpu_budget = effective_num_gpu_experts if scheduler_gpu_budget_per_layer is None else scheduler_gpu_budget_per_layer
        base_scheduler_config = SchedulerConfig(
            enabled=enable_dynamic_expert_scheduler,
            gpu_budget_per_layer=gpu_budget,
            hotness_decay=scheduler_hotness_decay,
            offload_tier=default_offload_tier,
            prefill_force_gpu_budget_per_layer=(
                gpu_budget
                if scheduler_prefill_force_gpu_budget_per_layer is None
                else scheduler_prefill_force_gpu_budget_per_layer
            ),
            prefill_offload_threshold_tokens=scheduler_prefill_offload_threshold_tokens,
            decode_promote_k=scheduler_decode_promote_k,
            prefill_collect_only=True,
            step_stride_prefill=8,
            step_stride_decode=1,
            demotion_idle_steps=0,
            migration_cooldown_steps=0,
            decode_require_prefetch_ready=False,
            prefetch_candidate_budget_per_layer=0,
        )
        normalized_scheduler_profile = (
            scheduler_profile.strip().lower().replace("-", "_")
            if scheduler_profile
            else SCHEDULER_PROFILE_BASELINE
        )
        profile_scheduler_config = resolve_scheduler_profile(
            normalized_scheduler_profile,
            base_config=base_scheduler_config,
        )
        resolved_scheduler_config = apply_scheduler_overrides(
            profile_scheduler_config,
            prefill_collect_only=scheduler_prefill_collect_only,
            step_stride_prefill=scheduler_step_stride_prefill,
            step_stride_decode=scheduler_step_stride_decode,
            demotion_idle_steps=scheduler_demotion_idle_steps,
            migration_cooldown_steps=scheduler_migration_cooldown_steps,
            decode_require_prefetch_ready=scheduler_decode_require_prefetch_ready,
            prefetch_candidate_budget_per_layer=scheduler_prefetch_candidate_budget_per_layer,
        )
        self.scheduler_profile = normalized_scheduler_profile
        self.dynamic_expert_scheduler = DynamicExpertScheduler(
            residency_plan=residency_plan,
            config=resolved_scheduler_config,
        )
        prepared_cache_budget = (
            scheduler_prepared_cache_budget_per_layer
            if scheduler_prepared_cache_budget_per_layer is not None
            else resolve_prepared_cache_budget(
                self.scheduler_profile,
                self.dynamic_expert_scheduler.config,
            )
        )
        self.prepared_cache_budget = int(prepared_cache_budget)
        self.prepared_controller_aggressiveness = resolve_prepared_controller_aggressiveness(
            self.scheduler_profile
        )
        self.enable_background_offload_worker = bool(enable_background_offload_worker)
        self.background_offload_poll_interval_seconds = float(background_offload_poll_interval_seconds)

        # DEBUG: 仅测试一层以排查崩溃原因
        # config.num_hidden_layers = 1
        
        # print(f"Instantiating Hybrid MoE model on {device}...")
        model_dtype = torch.bfloat16 if device == "cpu" or device.startswith("cuda") else torch.float32
        self.model = MixtralForCausalLM(
            config,
            self.dynamic_expert_scheduler.residency_plan.gpu_masks(),
            weight_path=model_path,
            offload_backend=offload_backend,
            offload_backend_kwargs=self.offload_backend_kwargs,
            residency_plan=self.dynamic_expert_scheduler.residency_plan,
            dynamic_expert_scheduler=self.dynamic_expert_scheduler,
            expert_prepared_cache_size=prepared_cache_budget,
            prepared_controller_aggressiveness=self.prepared_controller_aggressiveness,
            enable_background_offload_worker=self.enable_background_offload_worker,
            background_offload_poll_interval_seconds=self.background_offload_poll_interval_seconds,
        )
        self.model = self.model.to(device=device, dtype=model_dtype)
            
        # print(f"Loading weights from {model_path} into Python model...")
        load_model(self.model, model_path)
        
        # print("Initializing SimpleEngine...")
        self.engine = SimpleEngine(self.model, max_seq_len=max_seq_len, chunk_size=chunk_size)
        # print("Model loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # 1. Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        seq_len = input_ids.shape[1]
        print(f"Prompt tokens: {seq_len}. Starts generation...")

        try:
            self.engine.start_background_offload_worker()
            # 2. Prefill
            logits = self.engine.prefill(input_ids)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)

            generated_ids = [next_token.item()]

            # 3. Decode Loop
            for i in range(max_new_tokens - 1):
                logits = self.engine.decode_step(next_token, seq_len + i)
                next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)

                token_id = next_token.item()
                generated_ids.append(token_id)

                # Simple stopping criteria
                if token_id == self.tokenizer.eos_token_id:
                    break

            # 4. Decode output
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return output_text
        finally:
            self.engine.stop_background_offload_worker()
            self.shutdown()

    def get_offload_diagnostics(self) -> dict:
        layers = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            hybrid_moe = getattr(layer, "hybrid_moe", None)
            if hybrid_moe is None:
                continue
            layers.append({"layer_idx": layer_idx, **hybrid_moe.diagnostics()})
        return {
            "offload_backend": self.offload_backend,
            "scheduler_profile": scheduler_profile_summary(
                self.scheduler_profile,
                self.dynamic_expert_scheduler.config,
            ),
            "prepared_cache_budget_heuristic": resolve_prepared_cache_budget(
                self.scheduler_profile,
                self.dynamic_expert_scheduler.config,
            ),
            "prepared_cache_budget": self.prepared_cache_budget,
            "prepared_controller_aggressiveness": self.prepared_controller_aggressiveness,
            "background_offload_worker_enabled": self.enable_background_offload_worker,
            "background_offload_poll_interval_seconds": self.background_offload_poll_interval_seconds,
            "offload_refresh": self.model.model.offload_refresh_diagnostics(),
            "dynamic_scheduler": self.dynamic_expert_scheduler.diagnostics(),
            "layer_count": len(layers),
            "layers": layers,
        }

    def reset_offload_diagnostics(self) -> None:
        runtime = getattr(self.model.model, "offload_runtime", None)
        if runtime is not None:
            runtime.tick_calls = 0
            runtime.background_ticks = 0
            runtime.background_work_items_total = 0
            runtime.prefetch_submitted_total = 0
            runtime.background_warm_prebuilt_total = 0
            runtime.background_activation_ready_total = 0
            runtime.background_activation_applied_total = 0
            runtime.background_apply_queue_enqueued_total = 0
            runtime.ready_polled_total = 0
            runtime.activation_ready_total = 0
            runtime.ready_applied_total = 0
            runtime.ready_deferred_total = 0
            runtime.apply_batch_count_total = 0
            runtime.apply_batch_experts_total = 0
            runtime.apply_batch_evictions_total = 0
            runtime.apply_batch_activated_total = 0
            runtime.apply_batch_warm_total = 0
            runtime.apply_batch_cold_total = 0
            runtime.layers_touched_total = 0
            runtime.background_ready_callback_total = 0
            runtime.last_phase = ""
        reset_worker_fn = getattr(self.model.model, "reset_offload_worker_diagnostics", None)
        if reset_worker_fn is not None:
            reset_worker_fn()
        for layer in self.model.model.layers:
            hybrid_moe = getattr(layer, "hybrid_moe", None)
            if hybrid_moe is None:
                continue
            hybrid_moe.prefetch_requested = 0
            hybrid_moe.prefetch_enqueued = 0
            hybrid_moe.prefetch_materialized = 0
            hybrid_moe.prefetch_candidate_scans = 0
            hybrid_moe.runtime_evictions = 0
            hybrid_moe.runtime_skipped_demotion_cooldown = 0
            hybrid_moe.runtime_deferred_for_prefetch = 0
            hybrid_moe.decode_prefetch_hits = 0
            hybrid_moe.decode_prefetch_misses = 0
            hybrid_moe.pipeline_ready_applied = 0
            hybrid_moe.pipeline_ready_deferred = 0
            hybrid_moe.pipeline_ticks = 0
            hybrid_moe.pipeline_prefetch_overlap_hits = 0
            hybrid_moe.pipeline_promotion_source_activated = 0
            hybrid_moe.pipeline_promotion_source_warm = 0
            hybrid_moe.pipeline_promotion_source_cold = 0
            hybrid_moe.pipeline_apply_batches = 0
            hybrid_moe.pipeline_apply_batch_experts = 0
            hybrid_moe.pipeline_apply_batch_evictions = 0
            hybrid_moe.pipeline_apply_batch_activated = 0
            hybrid_moe.pipeline_apply_batch_warm = 0
            hybrid_moe.pipeline_apply_batch_cold = 0
            hybrid_moe.prepared_cache_rebalance_evicted_warm = 0
            hybrid_moe.prepared_cache_rebalance_evicted_activated = 0
            hybrid_moe.prepared_cache_rebalance_demoted_to_warm = 0
            hybrid_moe.prepared_cache_rebalance_dropped_to_ready = 0
            hybrid_moe.prepared_cache_activation_stage_bonus = 0.5
            hybrid_moe.cold_promotion_penalty = 0.0
            hybrid_moe.prepared_cache_rebalance_pressure_ema = 0.0
            hybrid_moe.prepared_cache_rebalance_events_last_tick = 0
            hybrid_moe.prepared_cache_rebalance_events_prev_total = 0
            hybrid_moe.warm_cache_hits = 0
            hybrid_moe.warm_cache_stores = 0
            hybrid_moe.warm_cache_evictions = 0
            hybrid_moe.warm_cache_prebuilt = 0
            hybrid_moe.warm_cache_device_transfers = 0
            hybrid_moe.activated_cache_hits = 0
            hybrid_moe.activated_cache_stores = 0
            hybrid_moe.activated_cache_evictions = 0
            hybrid_moe.activation_submitted = 0
            hybrid_moe.activation_ready = 0
            hybrid_moe.activation_applied = 0
            hybrid_moe.background_activation_applied = 0
            hybrid_moe.apply_queue_evictions = 0
            hybrid_moe.apply_queue_enqueued = 0
            hybrid_moe.apply_queue_committed = 0
            hybrid_moe.apply_queue_pruned = 0
            hybrid_moe.background_apply_queue_enqueued = 0
            hybrid_moe.apply_commit_queue_enqueued = 0
            hybrid_moe.apply_commit_queue_pruned = 0
            hybrid_moe.background_apply_commit_queue_enqueued = 0
            hybrid_moe.apply_commit_batch_queue_enqueued = 0
            hybrid_moe.apply_commit_batch_queue_pruned = 0
            hybrid_moe.background_apply_commit_batch_queue_enqueued = 0
            hybrid_moe.apply_commit_ready_hits = 0
            hybrid_moe.apply_commit_ready_stores = 0
            hybrid_moe.apply_commit_ready_pruned = 0
            hybrid_moe.background_apply_commit_resolved = 0
            hybrid_moe.apply_queue_commit_batches = 0
            hybrid_moe.apply_queue_commit_experts = 0
            hybrid_moe.background_apply_commit_batches = 0
            hybrid_moe.background_apply_commit_experts = 0
            hybrid_moe.apply_commit_queue_evictions = 0
            hybrid_moe.apply_commit_batch_queue_evictions = 0
            hybrid_moe.apply_queue_pressure_ema = 0.0
            hybrid_moe.apply_queue_events_last_tick = 0
            hybrid_moe.apply_queue_events_prev_total = 0
            hybrid_moe.apply_commit_queue_pressure_ema = 0.0
            hybrid_moe.apply_commit_queue_events_last_tick = 0
            hybrid_moe.apply_commit_queue_events_prev_total = 0

    def shutdown(self) -> None:
        shutdown_fn = getattr(self.model.model, "shutdown_offload_worker", None)
        if shutdown_fn is not None:
            shutdown_fn()
