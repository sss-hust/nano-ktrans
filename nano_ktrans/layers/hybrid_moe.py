"""
HybridMoE: CPU/GPU 混合专家层。

这是 nano-ktrans 的核心创新模块。它将 MoE 的专家分为两类：
- GPU 专家：常驻 GPU 显存，使用 PyTorch 直接计算
- CPU 专家：存放在 CPU 内存，通过 kt-kernel 的 AMX/AVX 加速计算

两类专家的计算是 **并发** 的：
1. 先通过 CPUMoEBackend.submit_forward() 异步启动 CPU 计算
2. 然后在 GPU 上同步计算 GPU 专家
3. 最后调用 CPUMoEBackend.sync_forward() 等待 CPU 结果
4. 合并两者的输出

这种设计充分利用了 CPU 和 GPU 的计算资源，使得即使只有少量 GPU 显存，
也能运行大型 MoE 模型。
"""

import torch
from torch import nn
from typing import Optional, Dict
from collections import OrderedDict
from threading import RLock

from nano_ktrans.kernels.cpu_moe import CPUMoEBackend
from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager
from nano_ktrans.kernels.expert_migration import MigrationLifecycle
from nano_ktrans.kernels.offload_backend import normalize_offload_backend_name
from nano_ktrans.kernels.pim_moe import PIMMoEBackend
from nano_ktrans.layers.expert_mlp import build_expert_module, load_expert_weights
from nano_ktrans.scheduler import DynamicExpertScheduler
from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan
from nano_ktrans.utils.context import get_context


class HybridMoE(nn.Module):
    """
    CPU/GPU 混合 MoE 层。

    参数：
        num_experts:           专家总数 (e.g., 8)
        top_k:                 每个 token 选择的专家数 (e.g., 2)
        hidden_size:           隐藏层维度
        moe_intermediate_size: 专家 FFN 中间维度
        gpu_experts:           GPU 上的专家模块 (nn.ModuleDict)
        gpu_experts_mask:      布尔掩码, True 表示该专家在 GPU 上
        layer_idx:             层索引
        weight_path:           权重文件路径
        num_threads:           CPU 推理线程数
        numa_pools:            NUMA 子池数量
        method:                CPU 后端方法 ("AMXINT4", "AMXINT8")
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts: nn.ModuleDict,
        gpu_experts_mask: torch.Tensor,
        layer_idx: int,
        weight_path: str,
        num_threads: int = 16,
        numa_pools: int = 1,
        chunked_prefill_size: int = 512,
        method: str = "AMXINT4",
        offload_backend: str = "cpu",
        offload_backend_kwargs: Optional[Dict[str, object]] = None,
        residency_plan: Optional[ExpertResidencyPlan] = None,
        dynamic_expert_scheduler: Optional[DynamicExpertScheduler] = None,
        router_use_softmax: bool = False,
        normalize_topk_prob: bool = True,
        expert_key_template: Optional[str] = None,
        expert_proj_names: Optional[Dict[str, str]] = None,
        experts_are_packed: bool = False,
        hidden_act: str = "silu",
        expert_prefetch_cache_size: int = 8,
        expert_prefetch_workers: int = 1,
        expert_warm_cache_size: int = 4,
        expert_prepared_cache_size: Optional[int] = None,
        prepared_controller_aggressiveness: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.gpu_experts_mask = gpu_experts_mask
        self.gpu_experts = gpu_experts
        self.layer_idx = layer_idx
        self.dynamic_expert_scheduler = dynamic_expert_scheduler
        self.residency_plan = residency_plan
        self.router_use_softmax = router_use_softmax
        self.normalize_topk_prob = normalize_topk_prob
        self.experts_are_packed = experts_are_packed
        self.hidden_act = hidden_act
        self.weight_path = weight_path
        self.expert_key_template = (
            expert_key_template
            or "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight"
        )
        self.expert_proj_names = expert_proj_names
        self.has_cpu_experts = bool((~gpu_experts_mask.bool()).any().item())
        self.offload_backend_name = normalize_offload_backend_name(offload_backend)
        self.offload_backend_kwargs = offload_backend_kwargs or {}
        self.applied_migration_ops = 0
        self.last_applied_migration_phase = ""
        self.applied_migration_history: list[dict[str, object]] = []
        self.prefetch_requested = 0
        self.prefetch_enqueued = 0
        self.prefetch_materialized = 0
        self.prefetch_candidate_scans = 0
        self.runtime_evictions = 0
        self.runtime_skipped_demotion_cooldown = 0
        self.runtime_deferred_for_prefetch = 0
        self.decode_prefetch_hits = 0
        self.decode_prefetch_misses = 0
        self.pipeline_ready_applied = 0
        self.pipeline_ready_deferred = 0
        self.pipeline_ticks = 0
        self.pipeline_prefetch_overlap_hits = 0
        self.pipeline_promotion_source_activated = 0
        self.pipeline_promotion_source_warm = 0
        self.pipeline_promotion_source_cold = 0
        self.pipeline_apply_batches = 0
        self.pipeline_apply_batch_experts = 0
        self.pipeline_apply_batch_evictions = 0
        self.pipeline_apply_batch_activated = 0
        self.pipeline_apply_batch_warm = 0
        self.pipeline_apply_batch_cold = 0
        self.prepared_cache_rebalance_evicted_warm = 0
        self.prepared_cache_rebalance_evicted_activated = 0
        self.prepared_cache_rebalance_demoted_to_warm = 0
        self.prepared_cache_rebalance_dropped_to_ready = 0
        self.prepared_cache_activation_stage_bonus = 0.5
        self.cold_promotion_penalty = 0.0
        self.prepared_cache_rebalance_pressure_ema = 0.0
        self.prepared_cache_rebalance_events_last_tick = 0
        self.prepared_cache_rebalance_events_prev_total = 0
        self.expert_warm_cache_size = max(0, int(expert_warm_cache_size))
        self.expert_prepared_cache_size = (
            None if expert_prepared_cache_size is None else max(0, int(expert_prepared_cache_size))
        )
        self.prepared_controller_aggressiveness = max(0.0, float(prepared_controller_aggressiveness))
        self._pipeline_lock = RLock()
        self.warm_expert_cache: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.activated_expert_cache: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.apply_candidate_queue: "OrderedDict[str, object]" = OrderedDict()
        self.apply_commit_queue: "OrderedDict[str, object]" = OrderedDict()
        self.apply_commit_batch_queue: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.resident_commit_batch_queue: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.resident_commit_finalize_queue: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.resident_commit_ready_cache: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.resident_commit_apply_queue: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.resident_commit_finalize_ready_queue: "OrderedDict[str, list[tuple[object, dict[str, object]]]]" = OrderedDict()
        self.apply_commit_ready_cache: "OrderedDict[str, dict[str, object]]" = OrderedDict()
        self.warm_cache_hits = 0
        self.warm_cache_stores = 0
        self.warm_cache_evictions = 0
        self.warm_cache_prebuilt = 0
        self.warm_cache_device_transfers = 0
        self.activated_cache_hits = 0
        self.activated_cache_stores = 0
        self.activated_cache_evictions = 0
        self.activation_submitted = 0
        self.activation_ready = 0
        self.activation_applied = 0
        self.background_activation_applied = 0
        self.apply_queue_evictions = 0
        self.apply_queue_enqueued = 0
        self.apply_queue_committed = 0
        self.apply_queue_pruned = 0
        self.background_apply_queue_enqueued = 0
        self.apply_commit_queue_enqueued = 0
        self.apply_commit_queue_pruned = 0
        self.background_apply_commit_queue_enqueued = 0
        self.apply_commit_batch_queue_enqueued = 0
        self.apply_commit_batch_queue_batches = 0
        self.apply_commit_batch_queue_committed_batches = 0
        self.apply_commit_batch_queue_pruned = 0
        self.background_apply_commit_batch_queue_enqueued = 0
        self.background_apply_commit_batch_queue_committed_batches = 0
        self.background_apply_commit_batch_queue_prefinalized_batches = 0
        self.resident_commit_batch_queue_enqueued = 0
        self.resident_commit_batch_queue_batches = 0
        self.resident_commit_batch_queue_committed_batches = 0
        self.resident_commit_batch_queue_pruned = 0
        self.resident_commit_batch_queue_evictions = 0
        self.background_resident_commit_batch_queue_enqueued = 0
        self.background_resident_commit_batch_queue_committed_batches = 0
        self.background_resident_commit_batch_queue_prefinalized_batches = 0
        self.resident_commit_finalize_queue_enqueued = 0
        self.resident_commit_finalize_queue_batches = 0
        self.resident_commit_finalize_queue_committed_batches = 0
        self.resident_commit_finalize_queue_pruned = 0
        self.resident_commit_finalize_queue_evictions = 0
        self.background_resident_commit_finalize_queue_enqueued = 0
        self.background_resident_commit_finalize_queue_committed_batches = 0
        self.background_resident_commit_finalize_queue_prefinalized_batches = 0
        self.resident_commit_ready_cache_stores = 0
        self.resident_commit_ready_cache_hits = 0
        self.resident_commit_ready_cache_pruned = 0
        self.resident_commit_ready_cache_evictions = 0
        self.background_resident_commit_ready_cache_stores = 0
        self.resident_commit_apply_queue_enqueued = 0
        self.resident_commit_apply_queue_batches = 0
        self.resident_commit_apply_queue_committed_batches = 0
        self.resident_commit_apply_queue_pruned = 0
        self.resident_commit_apply_queue_evictions = 0
        self.background_resident_commit_apply_queue_enqueued = 0
        self.background_resident_commit_apply_queue_committed_batches = 0
        self.background_resident_commit_apply_queue_prefinalized_batches = 0
        self.resident_commit_finalize_ready_queue_enqueued = 0
        self.resident_commit_finalize_ready_queue_batches = 0
        self.resident_commit_finalize_ready_queue_committed_batches = 0
        self.resident_commit_finalize_ready_queue_pruned = 0
        self.resident_commit_finalize_ready_queue_evictions = 0
        self.background_resident_commit_finalize_ready_queue_enqueued = 0
        self.background_resident_commit_finalize_ready_queue_committed_batches = 0
        self.background_resident_commit_finalize_ready_queue_prefinalized_batches = 0
        self.apply_commit_ready_hits = 0
        self.apply_commit_ready_stores = 0
        self.apply_commit_ready_pruned = 0
        self.background_apply_commit_resolved = 0
        self.apply_queue_commit_batches = 0
        self.apply_queue_commit_experts = 0
        self.background_apply_commit_batches = 0
        self.background_apply_commit_experts = 0
        self.apply_commit_queue_evictions = 0
        self.apply_commit_batch_queue_evictions = 0
        self.apply_queue_pressure_ema = 0.0
        self.apply_queue_events_last_tick = 0
        self.apply_queue_events_prev_total = 0
        self.apply_commit_queue_pressure_ema = 0.0
        self.apply_commit_queue_events_last_tick = 0
        self.apply_commit_queue_events_prev_total = 0
        self.apply_commit_batch_queue_pressure_ema = 0.0
        self.apply_commit_batch_queue_events_last_tick = 0
        self.apply_commit_batch_queue_events_prev_total = 0
        self.materialization_manager = ExpertMaterializationManager(
            weight_path=weight_path,
            expert_key_template=self.expert_key_template,
            expert_proj_names=self.expert_proj_names,
            max_cached_experts=expert_prefetch_cache_size,
            prefetch_workers=expert_prefetch_workers,
        )
        self.materialization_manager.set_ready_callback(self._on_materialization_ready)

        # 只有存在离线 CPU 专家时才初始化 CPU backend。
        self.offload_backend = None
        if self.has_cpu_experts:
            backend_kwargs = dict(
                layer_idx=layer_idx,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                gpu_experts_mask=gpu_experts_mask,
                weight_path=weight_path,
                num_threads=num_threads,
                numa_pools=numa_pools,
                chunked_prefill_size=chunked_prefill_size,
                method=method,
                expert_key_template=expert_key_template,
                expert_proj_names=expert_proj_names,
            )
            backend_kwargs.update(self.offload_backend_kwargs)
            if self.offload_backend_name in {"pim", "pim_shadow"}:
                backend_kwargs.setdefault(
                    "pim_execution_mode",
                    "real" if self.offload_backend_name == "pim" else "shadow",
                )
                self.offload_backend = PIMMoEBackend(**backend_kwargs)
            else:
                self.offload_backend = CPUMoEBackend(**backend_kwargs)

    def _build_runtime_expert(self, expert_idx: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        expert = build_expert_module(
            hidden_size=self.hidden_size,
            intermediate_size=self.offload_backend.intermediate_size if self.offload_backend is not None else 0,
            hidden_act=self.hidden_act,
            experts_are_packed=self.experts_are_packed,
        )
        expert = expert.to(device=device, dtype=dtype)
        weights = self.materialization_manager.get_expert(
            self.layer_idx,
            expert_idx,
        )
        load_expert_weights(expert, weights)
        expert.eval()
        return expert

    def _request_prefetch(self, expert_idx: int, *, allow_immediate_ready: bool = True) -> None:
        self.prefetch_requested += 1
        if self.offload_backend is not None:
            resident_weights = self.offload_backend.export_expert_weights(int(expert_idx))
            if resident_weights is not None:
                submitted = self.materialization_manager.stage_expert(
                    self.layer_idx,
                    int(expert_idx),
                    resident_weights,
                )
                if submitted:
                    self.prefetch_enqueued += 1
                    if allow_immediate_ready:
                        self.offload_backend.migration_manager.mark_state(
                            self.layer_idx,
                            expert_idx,
                            state=MigrationLifecycle.READY,
                        )
                    return
        submitted = self.materialization_manager.prefetch(self.layer_idx, expert_idx)
        if self.offload_backend is not None and submitted:
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                expert_idx,
                state=MigrationLifecycle.PREFETCHING,
            )
        if submitted:
            self.prefetch_enqueued += 1

    def _on_materialization_ready(self, layer_idx: int, expert_idx: int) -> None:
        if self.offload_backend is None:
            return
        if int(layer_idx) != int(self.layer_idx):
            return
        self.offload_backend.migration_manager.mark_state(
            self.layer_idx,
            int(expert_idx),
            state=MigrationLifecycle.READY,
        )

    def _prime_pending_promotions(self, *, phase: str) -> int:
        if (
            self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
        ):
            return 0
        queued_ops = self.offload_backend.migration_manager.peek_layer(self.layer_idx)
        if not queued_ops:
            return 0

        submitted = 0
        submit_limit = self._adaptive_prefetch_pending_limit(phase=phase)
        require_prefetch_ready = (
            phase == "decode"
            and self.dynamic_expert_scheduler.config.decode_require_prefetch_ready
        )
        for op in queued_ops:
            if op.dst != ExpertResidency.GPU:
                continue
            expert_idx = int(op.expert_idx)
            state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if state == MigrationLifecycle.APPLIED:
                continue
            is_ready = self.materialization_manager.is_ready(self.layer_idx, expert_idx)
            if is_ready:
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.READY,
                )
                continue
            if state != MigrationLifecycle.PREFETCHING and submitted < submit_limit:
                before = self.prefetch_enqueued
                self._request_prefetch(
                    expert_idx,
                    allow_immediate_ready=not require_prefetch_ready,
                )
                if self.prefetch_enqueued > before:
                    submitted += 1
            if require_prefetch_ready:
                if state != MigrationLifecycle.DEFERRED:
                    self.runtime_deferred_for_prefetch += 1
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.DEFERRED,
                )
        return submitted

    def refresh_offload_state(self) -> int:
        with self._pipeline_lock:
            if self.offload_backend is None:
                return 0
            callback_ready = self.materialization_manager.drain_ready_callbacks()
            if not self.materialization_manager.has_pending_or_ready():
                return int(callback_ready)
            ready_keys = self.materialization_manager.poll_ready()
            for layer_idx, expert_idx in ready_keys:
                if int(layer_idx) != int(self.layer_idx):
                    continue
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    int(expert_idx),
                    state=MigrationLifecycle.READY,
                )
            return int(callback_ready) + len(ready_keys)

    def background_tick_offload_state(self) -> int:
        with self._pipeline_lock:
            if self.offload_backend is None:
                return 0
            return int(self.materialization_manager.drain_ready_callbacks())

    def background_advance_offload_pipeline(
        self,
        *,
        phase: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, int]:
        with self._pipeline_lock:
            ready_polled = self.background_tick_offload_state()
            warm_prebuilt = 0
            activation_ready = 0
            activation_applied = 0
            apply_queue_enqueued = 0
            prev_background_apply_commit_queue_enqueued = self.background_apply_commit_queue_enqueued
            prev_background_apply_commit_batch_queue_enqueued = self.background_apply_commit_batch_queue_enqueued
            prev_background_apply_commit_batch_queue_prefinalized = (
                self.background_apply_commit_batch_queue_prefinalized_batches
            )
            prev_background_resident_commit_batch_queue_enqueued = (
                self.background_resident_commit_batch_queue_enqueued
            )
            prev_background_resident_commit_batch_queue_prefinalized = (
                self.background_resident_commit_batch_queue_prefinalized_batches
            )
            prev_background_resident_commit_finalize_queue_enqueued = (
                self.background_resident_commit_finalize_queue_enqueued
            )
            prev_background_resident_commit_finalize_queue_prefinalized = (
                self.background_resident_commit_finalize_queue_prefinalized_batches
            )
            prev_background_resident_commit_ready_cache_stores = (
                self.background_resident_commit_ready_cache_stores
            )
            prev_background_resident_commit_apply_queue_enqueued = (
                self.background_resident_commit_apply_queue_enqueued
            )
            prev_background_resident_commit_finalize_ready_queue_enqueued = (
                self.background_resident_commit_finalize_ready_queue_enqueued
            )
            background_apply_commit_batch_queue_enqueued = 0
            resident_commit_batch_queue_enqueued = 0
            resident_commit_batch_queue_prefinalized = 0
            resident_commit_finalize_queue_enqueued = 0
            resident_commit_finalize_queue_prefinalized = 0
            resident_commit_ready_cache_stores = 0
            resident_commit_apply_queue_enqueued = 0
            resident_commit_finalize_ready_queue_enqueued = 0
            if phase == "decode":
                preexisting_apply_commit_batch_keys = set(self.apply_commit_batch_queue.keys())
                preexisting_resident_commit_batch_keys = set(self.resident_commit_batch_queue.keys())
                preexisting_resident_commit_finalize_keys = set(self.resident_commit_finalize_queue.keys())
                preexisting_resident_commit_ready_keys = set(self.resident_commit_ready_cache.keys())
                preexisting_resident_commit_apply_keys = set(self.resident_commit_apply_queue.keys())
                preexisting_resident_commit_finalize_ready_keys = set(
                    self.resident_commit_finalize_ready_queue.keys()
                )
                warm_prebuilt = self._prebuild_ready_experts(phase=phase, device=device, dtype=dtype)
                activation_ready = self._activate_warmed_experts(phase=phase, device=device, dtype=dtype)
                apply_queue_enqueued = self._enqueue_activated_apply_candidates(phase=phase)
                self.background_apply_queue_enqueued += apply_queue_enqueued
                eligible_apply_candidate_ids = {
                    int(expert_idx) for expert_idx in self.apply_candidate_queue.keys()
                }
                self._stage_apply_commit_batch(
                    phase=phase,
                    active_experts=set(),
                    eligible_expert_ids=eligible_apply_candidate_ids,
                    max_commits=self._adaptive_apply_commit_limit(background=True),
                    background=True,
                )
                self._resolve_apply_commit_queue(
                    device=device,
                    dtype=dtype,
                    eligible_expert_ids=eligible_apply_candidate_ids,
                    max_resolves=self._adaptive_apply_commit_limit(background=True),
                    background=True,
                )
                self._stage_apply_commit_batch_queue(
                    eligible_expert_ids=eligible_apply_candidate_ids,
                    max_commits=self._adaptive_apply_commit_limit(background=True),
                    background=True,
                )
                background_apply_commit_batch_queue_enqueued = int(
                    self.background_apply_commit_batch_queue_enqueued
                    - prev_background_apply_commit_batch_queue_enqueued
                )
                self.background_apply_commit_batch_queue_prefinalized_batches += sum(
                    1
                    for batch_key in self.apply_commit_batch_queue.keys()
                    if batch_key not in preexisting_apply_commit_batch_keys
                )
                self._stage_resident_commit_batches(
                    eligible_batch_keys=None,
                    max_batches=self._adaptive_apply_commit_batch_limit(background=True),
                    background=True,
                )
                resident_commit_batch_queue_enqueued = int(
                    self.background_resident_commit_batch_queue_enqueued
                    - prev_background_resident_commit_batch_queue_enqueued
                )
                self.background_resident_commit_batch_queue_prefinalized_batches += sum(
                    1
                    for batch_key in self.resident_commit_batch_queue.keys()
                    if batch_key not in preexisting_resident_commit_batch_keys
                )
                resident_commit_batch_queue_prefinalized = int(
                    self.background_resident_commit_batch_queue_prefinalized_batches
                    - prev_background_resident_commit_batch_queue_prefinalized
                )
                self._stage_resident_commit_finalize_queue(
                    eligible_batch_keys=None,
                    max_batches=self._adaptive_apply_commit_batch_limit(background=True),
                    background=True,
                )
                resident_commit_finalize_queue_enqueued = int(
                    self.background_resident_commit_finalize_queue_enqueued
                    - prev_background_resident_commit_finalize_queue_enqueued
                )
                self.background_resident_commit_finalize_queue_prefinalized_batches += sum(
                    1
                    for batch_key in self.resident_commit_finalize_queue.keys()
                    if batch_key not in preexisting_resident_commit_finalize_keys
                )
                resident_commit_finalize_queue_prefinalized = int(
                    self.background_resident_commit_finalize_queue_prefinalized_batches
                    - prev_background_resident_commit_finalize_queue_prefinalized
                )
                resident_commit_ready_cache_stores = self._stage_resident_commit_ready_cache(
                    eligible_batch_keys=None,
                    max_batches=self._adaptive_apply_commit_batch_limit(background=True),
                    background=True,
                )
                resident_commit_apply_queue_enqueued = self._stage_resident_commit_apply_queue(
                    eligible_batch_keys=None,
                    max_batches=self._adaptive_apply_commit_batch_limit(background=True),
                    background=True,
                )
                resident_commit_finalize_ready_queue_enqueued = (
                    self._stage_resident_commit_finalize_ready_queue(
                        eligible_batch_keys=None,
                        max_batches=self._adaptive_apply_commit_batch_limit(background=True),
                        background=True,
                    )
                )
                activation_applied = self._background_apply_activated_experts(
                    phase=phase,
                    eligible_batch_keys=preexisting_resident_commit_finalize_ready_keys,
                    stage_resident_batches=False,
                )
            return {
                "ready_polled": int(ready_polled),
                "warm_prebuilt": int(warm_prebuilt),
                "activation_ready": int(activation_ready),
                "activation_applied": int(activation_applied),
                "apply_queue_enqueued": int(apply_queue_enqueued),
                "apply_commit_queue_enqueued": int(
                    self.background_apply_commit_queue_enqueued - prev_background_apply_commit_queue_enqueued
                ),
                "apply_commit_batch_queue_enqueued": background_apply_commit_batch_queue_enqueued,
                "apply_commit_batch_queue_prefinalized": int(
                    self.background_apply_commit_batch_queue_prefinalized_batches
                    - prev_background_apply_commit_batch_queue_prefinalized
                ),
                "resident_commit_batch_queue_enqueued": resident_commit_batch_queue_enqueued,
                "resident_commit_batch_queue_prefinalized": resident_commit_batch_queue_prefinalized,
                "resident_commit_finalize_queue_enqueued": resident_commit_finalize_queue_enqueued,
                "resident_commit_finalize_queue_prefinalized": resident_commit_finalize_queue_prefinalized,
                "resident_commit_ready_cache_stores": int(
                    self.background_resident_commit_ready_cache_stores
                    - prev_background_resident_commit_ready_cache_stores
                ) if phase == "decode" else 0,
                "resident_commit_apply_queue_enqueued": int(
                    self.background_resident_commit_apply_queue_enqueued
                    - prev_background_resident_commit_apply_queue_enqueued
                ) if phase == "decode" else 0,
                "resident_commit_finalize_ready_queue_enqueued": int(
                    self.background_resident_commit_finalize_ready_queue_enqueued
                    - prev_background_resident_commit_finalize_ready_queue_enqueued
                ) if phase == "decode" else 0,
            }

    def _insert_warm_module(self, expert_idx: int, module: nn.Module) -> None:
        expert_key = str(expert_idx)
        self.warm_expert_cache[expert_key] = module.to(device="cpu")
        self.warm_expert_cache.move_to_end(expert_key)

    def _store_warm_module(self, expert_idx: int, module: nn.Module, *, count_store: bool) -> None:
        if self.expert_warm_cache_size <= 0:
            return
        self._insert_warm_module(expert_idx, module)
        if count_store:
            self.warm_cache_stores += 1
        self._rebalance_prepared_caches()

    def _effective_warm_cache_limit(self) -> int:
        limit = self.expert_warm_cache_size
        effective_prepared_limit = self._effective_prepared_cache_limit()
        if effective_prepared_limit is None:
            return limit
        remaining_budget = max(0, effective_prepared_limit - len(self.activated_expert_cache))
        return min(limit, remaining_budget)

    def _prepared_cache_size(self) -> int:
        return len(self.warm_expert_cache) + len(self.activated_expert_cache)

    def _prepared_cache_rebalance_pressure(self) -> float:
        if not self.expert_prepared_cache_size:
            return 0.0
        rebalance_events = (
            self.prepared_cache_rebalance_evicted_warm
            + self.prepared_cache_rebalance_evicted_activated
        )
        normalization = self.pipeline_ticks if self.pipeline_ticks > 0 else int(self.expert_prepared_cache_size)
        return rebalance_events / max(1, int(normalization))

    def _prepared_cache_rebalance_pressure_step(self) -> float:
        if not self.expert_prepared_cache_size:
            return 0.0
        return self.prepared_cache_rebalance_events_last_tick / max(1, int(self.expert_prepared_cache_size))

    def _update_prepared_cache_rebalance_pressure_ema(self) -> None:
        step_pressure = self._prepared_cache_rebalance_pressure_step()
        self.prepared_cache_rebalance_pressure_ema = (
            0.8 * self.prepared_cache_rebalance_pressure_ema
        ) + (0.2 * step_pressure)

    def _update_prepared_cache_rebalance_pressure_signals(self) -> None:
        current_total = (
            self.prepared_cache_rebalance_evicted_warm
            + self.prepared_cache_rebalance_evicted_activated
        )
        self.prepared_cache_rebalance_events_last_tick = max(
            0,
            current_total - self.prepared_cache_rebalance_events_prev_total,
        )
        self.prepared_cache_rebalance_events_prev_total = current_total
        self._update_prepared_cache_rebalance_pressure_ema()

    def _prepared_cache_budget_backoff(self) -> int:
        if self.expert_prepared_cache_size is None:
            return 0
        base_limit = int(self.expert_prepared_cache_size)
        if base_limit <= 1:
            return 0

        pressure = max(
            self._prepared_cache_rebalance_pressure_step(),
            self._prepared_cache_rebalance_pressure(),
            self.prepared_cache_rebalance_pressure_ema,
        )
        backoff = 0
        if pressure >= 1.0:
            backoff += 1
        if pressure >= 2.0:
            backoff += 1

        if self.prepared_cache_activation_stage_bonus > 0.25 and backoff > 0:
            backoff -= 1
        if self.prepared_cache_activation_stage_bonus >= 1.0 and backoff > 0:
            backoff -= 1

        if self.cold_promotion_penalty >= 1.0 and backoff > 0:
            backoff -= 1
        if self.cold_promotion_penalty >= 1.5 and backoff > 0:
            backoff -= 1

        return min(max(0, backoff), max(0, base_limit - 1))

    def _effective_prepared_cache_limit(self) -> Optional[int]:
        if self.expert_prepared_cache_size is None:
            return None
        base_limit = int(self.expert_prepared_cache_size)
        if base_limit <= 1:
            return base_limit
        return max(1, base_limit - self._prepared_cache_budget_backoff())

    def _prepared_cache_retention_score(self, expert_idx: int, cache_kind: str) -> float:
        stage_bonus = self.prepared_cache_activation_stage_bonus if cache_kind == "activated" else 0.0
        return self._hotness_score(expert_idx) + stage_bonus

    def _update_prepared_cache_stage_bonus(
        self,
        *,
        activated_demotions: int,
        warm_drops: int,
    ) -> None:
        if activated_demotions > 0:
            self.prepared_cache_activation_stage_bonus = min(
                2.0,
                self.prepared_cache_activation_stage_bonus + (0.25 * activated_demotions),
            )
        elif warm_drops > 0:
            self.prepared_cache_activation_stage_bonus = max(
                0.0,
                self.prepared_cache_activation_stage_bonus - (0.25 * warm_drops),
            )

    def _pick_prepared_cache_victim(self) -> tuple[str, str] | None:
        candidates: list[tuple[str, str, float, int, int]] = []
        for order, expert_key in enumerate(self.warm_expert_cache.keys()):
            expert_idx = int(expert_key)
            candidates.append(
                (
                    "warm",
                    expert_key,
                    self._prepared_cache_retention_score(expert_idx, "warm"),
                    self._migration_state_priority(expert_idx),
                    order,
                )
            )
        for order, expert_key in enumerate(self.activated_expert_cache.keys()):
            expert_idx = int(expert_key)
            candidates.append(
                (
                    "activated",
                    expert_key,
                    self._prepared_cache_retention_score(expert_idx, "activated"),
                    self._migration_state_priority(expert_idx),
                    order,
                )
            )
        if not candidates:
            return None
        cache_kind, expert_key, *_ = min(
            candidates,
            key=lambda item: (item[2], item[3], item[4]),
        )
        return cache_kind, expert_key

    def _trim_warm_cache_to_budget(self) -> None:
        limit = self._effective_warm_cache_limit()
        while len(self.warm_expert_cache) > limit:
            evicted_key = self._pick_warm_cache_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.warm_expert_cache.popitem(last=False)
            else:
                self.warm_expert_cache.pop(evicted_key)
            if self.offload_backend is not None:
                evicted_idx = int(evicted_key)
                state = self.offload_backend.migration_manager.state_for(self.layer_idx, evicted_idx)
                if state == MigrationLifecycle.WARMED:
                    self.offload_backend.migration_manager.mark_state(
                        self.layer_idx,
                        evicted_idx,
                        state=MigrationLifecycle.READY,
                    )
            self.warm_cache_evictions += 1

    def _rebalance_prepared_caches(self) -> None:
        effective_prepared_limit = self._effective_prepared_cache_limit()
        if effective_prepared_limit is None:
            self._trim_warm_cache_to_budget()
            return

        activated_demotions = 0
        warm_drops = 0
        while self._prepared_cache_size() > effective_prepared_limit:
            victim = self._pick_prepared_cache_victim()
            if victim is None:
                break
            cache_kind, expert_key = victim
            expert_idx = int(expert_key)
            if cache_kind == "activated":
                module = self.activated_expert_cache.pop(expert_key)
                if self.offload_backend is not None:
                    state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                    if state == MigrationLifecycle.ACTIVATED:
                        self.offload_backend.migration_manager.mark_state(
                            self.layer_idx,
                            expert_idx,
                            state=MigrationLifecycle.WARMED,
                        )
                self.activated_cache_evictions += 1
                self.prepared_cache_rebalance_evicted_activated += 1
                self.prepared_cache_rebalance_demoted_to_warm += 1
                activated_demotions += 1
                self._insert_warm_module(expert_idx, module)
                continue
            self.warm_expert_cache.pop(expert_key)
            if self.offload_backend is not None:
                state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if state == MigrationLifecycle.WARMED:
                    self.offload_backend.migration_manager.mark_state(
                        self.layer_idx,
                        expert_idx,
                        state=MigrationLifecycle.READY,
                    )
            self.warm_cache_evictions += 1
            self.prepared_cache_rebalance_evicted_warm += 1
            self.prepared_cache_rebalance_dropped_to_ready += 1
            warm_drops += 1

        self._update_prepared_cache_stage_bonus(
            activated_demotions=activated_demotions,
            warm_drops=warm_drops,
        )
        self._trim_warm_cache_to_budget()

    def _pick_warm_cache_victim_key(self) -> str | None:
        if not self.warm_expert_cache:
            return None
        ordered_keys = list(self.warm_expert_cache.keys())
        return min(
            ordered_keys,
            key=lambda expert_key: (
                self._migration_state_priority(int(expert_key)),
                self._hotness_score(int(expert_key)),
                ordered_keys.index(expert_key),
            ),
        )

    def _activated_cache_limit(self) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 1
        return max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))

    def _apply_queue_limit(self) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 1
        limit = max(
            1,
            int(self.dynamic_expert_scheduler.config.decode_promote_k),
        )
        if self.prepared_controller_aggressiveness >= 0.5:
            limit += 1
        if self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        return limit

    def _pick_apply_queue_victim_key(self) -> str | None:
        if not self.apply_candidate_queue:
            return None
        ordered_keys = list(self.apply_candidate_queue.keys())
        return min(
            ordered_keys,
            key=lambda expert_key: (
                self._hotness_score(int(expert_key)),
                self._migration_state_priority(int(expert_key)),
                ordered_keys.index(expert_key),
            ),
        )

    def _apply_commit_queue_limit(self) -> int:
        base_limit = self._apply_queue_limit()
        if self.prepared_controller_aggressiveness >= 1.0:
            base_limit += 1
        return max(1, base_limit)

    def _apply_commit_batch_queue_limit(self) -> int:
        base_limit = self._apply_commit_queue_limit()
        return max(1, base_limit)

    def _resident_commit_batch_queue_limit(self) -> int:
        base_limit = self._apply_commit_batch_queue_limit()
        return max(1, base_limit)

    def _resident_commit_finalize_queue_limit(self) -> int:
        base_limit = self._resident_commit_batch_queue_limit()
        return max(1, base_limit)

    def _resident_commit_ready_cache_limit(self) -> int:
        base_limit = self._resident_commit_finalize_queue_limit()
        return max(1, base_limit)

    def _resident_commit_apply_queue_limit(self) -> int:
        base_limit = self._resident_commit_ready_cache_limit()
        return max(1, base_limit)

    def _resident_commit_finalize_ready_queue_limit(self) -> int:
        base_limit = self._resident_commit_apply_queue_limit()
        return max(1, base_limit)

    def _pick_apply_commit_queue_victim_key(self) -> str | None:
        if not self.apply_commit_queue:
            return None
        ordered_keys = list(self.apply_commit_queue.keys())
        return min(
            ordered_keys,
            key=lambda expert_key: (
                self._hotness_score(int(expert_key)),
                self._migration_state_priority(int(expert_key)),
                ordered_keys.index(expert_key),
            ),
        )

    def _pick_apply_commit_batch_queue_victim_key(self) -> str | None:
        if not self.apply_commit_batch_queue:
            return None
        ordered_keys = list(self.apply_commit_batch_queue.keys())
        return min(
            ordered_keys,
            key=lambda batch_key: (
                max(
                    self._hotness_score(int(op.expert_idx))
                    for op, _resolved in self.apply_commit_batch_queue[batch_key]
                ),
                max(
                    self._migration_state_priority(int(op.expert_idx))
                    for op, _resolved in self.apply_commit_batch_queue[batch_key]
                ),
                ordered_keys.index(batch_key),
            ),
        )

    def _pick_resident_commit_batch_queue_victim_key(self) -> str | None:
        if not self.resident_commit_batch_queue:
            return None
        ordered_keys = list(self.resident_commit_batch_queue.keys())
        return min(
            ordered_keys,
            key=lambda batch_key: (
                max(
                    self._hotness_score(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_batch_queue[batch_key]
                ),
                max(
                    self._migration_state_priority(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_batch_queue[batch_key]
                ),
                ordered_keys.index(batch_key),
            ),
        )

    def _pick_resident_commit_finalize_queue_victim_key(self) -> str | None:
        if not self.resident_commit_finalize_queue:
            return None
        ordered_keys = list(self.resident_commit_finalize_queue.keys())
        return min(
            ordered_keys,
            key=lambda batch_key: (
                max(
                    self._hotness_score(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_finalize_queue[batch_key]
                ),
                max(
                    self._migration_state_priority(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_finalize_queue[batch_key]
                ),
                ordered_keys.index(batch_key),
            ),
        )

    def _pick_resident_commit_apply_queue_victim_key(self) -> str | None:
        if not self.resident_commit_apply_queue:
            return None
        ordered_keys = list(self.resident_commit_apply_queue.keys())
        return min(
            ordered_keys,
            key=lambda batch_key: (
                max(
                    self._hotness_score(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_apply_queue[batch_key]
                ),
                max(
                    self._migration_state_priority(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_apply_queue[batch_key]
                ),
                ordered_keys.index(batch_key),
            ),
        )

    def _pick_resident_commit_finalize_ready_queue_victim_key(self) -> str | None:
        if not self.resident_commit_finalize_ready_queue:
            return None
        ordered_keys = list(self.resident_commit_finalize_ready_queue.keys())
        return min(
            ordered_keys,
            key=lambda batch_key: (
                max(
                    self._hotness_score(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_finalize_ready_queue[batch_key]
                ),
                max(
                    self._migration_state_priority(int(op.expert_idx))
                    for op, _resolved in self.resident_commit_finalize_ready_queue[batch_key]
                ),
                ordered_keys.index(batch_key),
            ),
        )

    def _rebalance_apply_commit_queue(self) -> None:
        limit = self._apply_commit_queue_limit()
        while len(self.apply_commit_queue) > limit:
            evicted_key = self._pick_apply_commit_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.apply_commit_queue.popitem(last=False)
            else:
                self.apply_commit_queue.pop(evicted_key, None)
            self.apply_commit_queue_evictions += 1
            self.apply_commit_ready_cache.pop(evicted_key, None)
            stale_batches = [
                batch_key
                for batch_key, batch_entries in self.apply_commit_batch_queue.items()
                if any(str(int(op.expert_idx)) == evicted_key for op, _resolved in batch_entries)
            ]
            for batch_key in stale_batches:
                self.apply_commit_batch_queue.pop(batch_key, None)

    def _rebalance_apply_commit_batch_queue(self) -> None:
        limit = self._apply_commit_batch_queue_limit()
        while len(self.apply_commit_batch_queue) > limit:
            evicted_key = self._pick_apply_commit_batch_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.apply_commit_batch_queue.popitem(last=False)
            else:
                self.apply_commit_batch_queue.pop(evicted_key, None)
            self.apply_commit_batch_queue_evictions += 1

    def _rebalance_resident_commit_batch_queue(self) -> None:
        limit = self._resident_commit_batch_queue_limit()
        while len(self.resident_commit_batch_queue) > limit:
            evicted_key = self._pick_resident_commit_batch_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.resident_commit_batch_queue.popitem(last=False)
            else:
                self.resident_commit_batch_queue.pop(evicted_key, None)
            self.resident_commit_batch_queue_evictions += 1

    def _rebalance_resident_commit_finalize_queue(self) -> None:
        limit = self._resident_commit_finalize_queue_limit()
        while len(self.resident_commit_finalize_queue) > limit:
            evicted_key = self._pick_resident_commit_finalize_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.resident_commit_finalize_queue.popitem(last=False)
            else:
                self.resident_commit_finalize_queue.pop(evicted_key, None)
            self.resident_commit_finalize_queue_evictions += 1

    def _rebalance_resident_commit_ready_cache(self) -> None:
        limit = self._resident_commit_ready_cache_limit()
        while len(self.resident_commit_ready_cache) > limit:
            evicted_key, _ = self.resident_commit_ready_cache.popitem(last=False)
            self.resident_commit_ready_cache_evictions += 1

    def _rebalance_resident_commit_apply_queue(self) -> None:
        limit = self._resident_commit_apply_queue_limit()
        while len(self.resident_commit_apply_queue) > limit:
            evicted_key = self._pick_resident_commit_apply_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.resident_commit_apply_queue.popitem(last=False)
            else:
                self.resident_commit_apply_queue.pop(evicted_key, None)
            self.resident_commit_apply_queue_evictions += 1

    def _rebalance_resident_commit_finalize_ready_queue(self) -> None:
        limit = self._resident_commit_finalize_ready_queue_limit()
        while len(self.resident_commit_finalize_ready_queue) > limit:
            evicted_key = self._pick_resident_commit_finalize_ready_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.resident_commit_finalize_ready_queue.popitem(last=False)
            else:
                self.resident_commit_finalize_ready_queue.pop(evicted_key, None)
            self.resident_commit_finalize_ready_queue_evictions += 1

    def _prune_apply_commit_ready_cache(self) -> int:
        stale_keys: list[str] = []
        for expert_key in list(self.apply_commit_ready_cache.keys()):
            expert_idx = int(expert_key)
            lifecycle = None
            if self.offload_backend is not None:
                lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if (
                expert_key not in self.apply_commit_queue
                or bool(self.gpu_experts_mask[expert_idx].item())
                or lifecycle not in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
            ):
                stale_keys.append(expert_key)

        for expert_key in stale_keys:
            self.apply_commit_ready_cache.pop(expert_key, None)
        self.apply_commit_ready_pruned += len(stale_keys)
        return len(stale_keys)

    def _resolve_apply_commit_queue(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        eligible_expert_ids: Optional[set[int]] = None,
        max_resolves: Optional[int],
        background: bool,
    ) -> int:
        if not self.apply_commit_queue:
            return 0

        resolve_limit = max_resolves if max_resolves is not None else self._adaptive_apply_commit_limit(background=background)
        resolve_limit = max(1, int(resolve_limit))
        resolved = 0
        for expert_key, op in list(self.apply_commit_queue.items()):
            if expert_key in self.apply_commit_ready_cache:
                continue
            expert_idx = int(expert_key)
            if eligible_expert_ids is not None and expert_idx not in eligible_expert_ids:
                continue
            resolved_item = self._resolve_promotion_module(expert_idx, device, dtype)
            self.apply_commit_ready_cache[expert_key] = {
                "op": op,
                "resolved": resolved_item,
            }
            self.apply_commit_ready_stores += 1
            if background:
                self.background_apply_commit_resolved += 1
            resolved += 1
            if resolved >= resolve_limit:
                break
        return resolved

    def _stage_apply_commit_batch_queue(
        self,
        *,
        eligible_expert_ids: Optional[set[int]] = None,
        max_commits: Optional[int],
        background: bool,
    ) -> int:
        if not self.apply_commit_queue:
            return 0

        commit_limit = max_commits if max_commits is not None else self._adaptive_apply_commit_limit(background=background)
        commit_limit = max(1, int(commit_limit))
        candidate_ops = []
        for expert_key, op in list(self.apply_commit_queue.items()):
            expert_idx = int(expert_key)
            if eligible_expert_ids is not None and expert_idx not in eligible_expert_ids:
                continue
            if expert_key not in self.apply_commit_ready_cache:
                continue
            candidate_ops.append((op, self.apply_commit_ready_cache[expert_key]["resolved"]))

        if not candidate_ops:
            return 0

        candidate_ops.sort(key=lambda item: self._promotion_sort_key(item[0], set(), "decode"))
        selected_ops = candidate_ops[:commit_limit]
        batch_key = ",".join(str(int(op.expert_idx)) for op, _resolved in selected_ops)
        if not batch_key:
            return 0
        enqueued = 0
        was_present = batch_key in self.apply_commit_batch_queue
        self.apply_commit_batch_queue[batch_key] = selected_ops
        self.apply_commit_batch_queue.move_to_end(batch_key)
        if not was_present:
            self.apply_commit_batch_queue_enqueued += 1
            self.apply_commit_batch_queue_batches += 1
            enqueued += 1
            if background:
                self.background_apply_commit_batch_queue_enqueued += 1

        self._rebalance_apply_commit_batch_queue()
        return enqueued

    def _stage_resident_commit_batches(
        self,
        *,
        eligible_batch_keys: Optional[set[str]] = None,
        max_batches: Optional[int],
        background: bool,
    ) -> int:
        if not self.apply_commit_batch_queue:
            return 0

        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batches is None
            else max(1, int(max_batches))
        )
        candidate_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in list(self.apply_commit_batch_queue.items()):
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            if not batch_entries:
                continue
            candidate_batches.append((batch_key, batch_entries))

        if not candidate_batches:
            return 0

        candidate_batches.sort(
            key=lambda item: min(
                self._promotion_sort_key(op, set(), "decode")
                for op, _resolved in item[1]
            )
        )
        selected_batches = candidate_batches[:batch_limit]
        enqueued = 0
        for batch_key, batch_entries in selected_batches:
            was_present = batch_key in self.resident_commit_batch_queue
            self.resident_commit_batch_queue[batch_key] = batch_entries
            self.resident_commit_batch_queue.move_to_end(batch_key)
            if not was_present:
                self.resident_commit_batch_queue_enqueued += 1
                self.resident_commit_batch_queue_batches += 1
                enqueued += 1
                if background:
                    self.background_resident_commit_batch_queue_enqueued += 1

        self._rebalance_resident_commit_batch_queue()
        return enqueued

    def _rebalance_apply_candidate_queue(self) -> None:
        limit = self._apply_queue_limit()
        while len(self.apply_candidate_queue) > limit:
            evicted_key = self._pick_apply_queue_victim_key()
            if evicted_key is None:
                evicted_key, _ = self.apply_candidate_queue.popitem(last=False)
            else:
                self.apply_candidate_queue.pop(evicted_key, None)
            self.apply_queue_evictions += 1

    def _pick_activated_cache_victim_key(self) -> str | None:
        if not self.activated_expert_cache:
            return None
        ordered_keys = list(self.activated_expert_cache.keys())
        return min(
            ordered_keys,
            key=lambda expert_key: (
                self._migration_state_priority(int(expert_key)),
                self._hotness_score(int(expert_key)),
                ordered_keys.index(expert_key),
            ),
        )

    def _hotness_score(self, expert_idx: int) -> float:
        if self.residency_plan is None:
            return 0.0
        state = self.residency_plan.layer_state(self.layer_idx)
        if int(expert_idx) >= state.hotness.numel():
            return 0.0
        return float(state.hotness[int(expert_idx)].item())

    def _migration_state_priority(self, expert_idx: int) -> int:
        if self.offload_backend is None:
            return 0
        state = self.offload_backend.migration_manager.state_for(self.layer_idx, int(expert_idx))
        priorities = {
            MigrationLifecycle.ACTIVATED: 3,
            MigrationLifecycle.WARMED: 2,
            MigrationLifecycle.READY: 1,
        }
        return priorities.get(state, 0)

    def _prepared_cache_pressure(self) -> float:
        effective_prepared_limit = self._effective_prepared_cache_limit()
        if not effective_prepared_limit:
            return 0.0
        return self._prepared_cache_size() / max(1, int(effective_prepared_limit))

    def _prepared_controller_engaged(self) -> bool:
        return (
            self._prepared_cache_budget_backoff() > 0
            or self._prepared_cache_pressure() >= 1.0
            or self._apply_queue_budget_backoff() > 0
            or self._apply_queue_pressure() >= 1.0
            or self._apply_commit_queue_budget_backoff() > 0
            or self._apply_commit_queue_pressure() >= 1.0
            or self._apply_commit_batch_queue_budget_backoff() > 0
            or self._apply_commit_batch_queue_pressure() >= 1.0
        )

    def _apply_queue_pressure(self) -> float:
        limit = self._apply_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_candidate_queue) / max(1, int(limit))
        normalization = self.pipeline_ticks if self.pipeline_ticks > 0 else int(limit)
        eviction_pressure = self.apply_queue_evictions / max(1, int(normalization))
        return queue_utilization + eviction_pressure

    def _apply_queue_pressure_step(self) -> float:
        limit = self._apply_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_candidate_queue) / max(1, int(limit))
        return queue_utilization + (
            self.apply_queue_events_last_tick / max(1, int(limit))
        )

    def _update_apply_queue_pressure_ema(self) -> None:
        step_pressure = self._apply_queue_pressure_step()
        self.apply_queue_pressure_ema = (0.8 * self.apply_queue_pressure_ema) + (0.2 * step_pressure)

    def _update_apply_queue_pressure_signals(self) -> None:
        current_total = self.apply_queue_evictions + self.apply_commit_queue_evictions
        self.apply_queue_events_last_tick = max(
            0,
            current_total - self.apply_queue_events_prev_total,
        )
        self.apply_queue_events_prev_total = current_total
        self._update_apply_queue_pressure_ema()

    def _apply_queue_budget_backoff(self) -> int:
        limit = self._apply_queue_limit()
        if limit <= 1:
            return 0

        pressure = max(
            self._apply_queue_pressure_step(),
            self._apply_queue_pressure(),
            self.apply_queue_pressure_ema,
        )
        backoff = 0
        if pressure >= 1.0:
            backoff += 1
        if pressure >= 2.0:
            backoff += 1

        if self.cold_promotion_penalty >= 1.0 and backoff > 0:
            backoff -= 1
        if self.cold_promotion_penalty >= 1.5 and backoff > 0:
            backoff -= 1

        return min(max(0, backoff), max(0, limit - 1))

    def _apply_commit_queue_pressure(self) -> float:
        limit = self._apply_commit_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_commit_queue) / max(1, int(limit))
        normalization = self.pipeline_ticks if self.pipeline_ticks > 0 else int(limit)
        eviction_pressure = self.apply_commit_queue_evictions / max(1, int(normalization))
        return queue_utilization + eviction_pressure

    def _apply_commit_queue_pressure_step(self) -> float:
        limit = self._apply_commit_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_commit_queue) / max(1, int(limit))
        return queue_utilization + (
            self.apply_commit_queue_events_last_tick / max(1, int(limit))
        )

    def _update_apply_commit_queue_pressure_ema(self) -> None:
        step_pressure = self._apply_commit_queue_pressure_step()
        self.apply_commit_queue_pressure_ema = (0.8 * self.apply_commit_queue_pressure_ema) + (0.2 * step_pressure)

    def _update_apply_commit_queue_pressure_signals(self) -> None:
        current_total = self.apply_commit_queue_evictions
        self.apply_commit_queue_events_last_tick = max(
            0,
            current_total - self.apply_commit_queue_events_prev_total,
        )
        self.apply_commit_queue_events_prev_total = current_total
        self._update_apply_commit_queue_pressure_ema()

    def _apply_commit_queue_budget_backoff(self) -> int:
        limit = self._apply_commit_queue_limit()
        if limit <= 1:
            return 0

        pressure = max(
            self._apply_commit_queue_pressure_step(),
            self._apply_commit_queue_pressure(),
            self.apply_commit_queue_pressure_ema,
        )
        backoff = 0
        if pressure >= 1.0:
            backoff += 1
        if pressure >= 2.0:
            backoff += 1

        if self.cold_promotion_penalty >= 1.0 and backoff > 0:
            backoff -= 1
        if self.cold_promotion_penalty >= 1.5 and backoff > 0:
            backoff -= 1

        return min(max(0, backoff), max(0, limit - 1))

    def _apply_commit_batch_queue_pressure(self) -> float:
        limit = self._apply_commit_batch_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_commit_batch_queue) / max(1, int(limit))
        normalization = self.pipeline_ticks if self.pipeline_ticks > 0 else int(limit)
        eviction_pressure = self.apply_commit_batch_queue_evictions / max(1, int(normalization))
        return queue_utilization + eviction_pressure

    def _apply_commit_batch_queue_pressure_step(self) -> float:
        limit = self._apply_commit_batch_queue_limit()
        if limit <= 0:
            return 0.0
        queue_utilization = len(self.apply_commit_batch_queue) / max(1, int(limit))
        return queue_utilization + (
            self.apply_commit_batch_queue_events_last_tick / max(1, int(limit))
        )

    def _update_apply_commit_batch_queue_pressure_ema(self) -> None:
        step_pressure = self._apply_commit_batch_queue_pressure_step()
        self.apply_commit_batch_queue_pressure_ema = (
            0.8 * self.apply_commit_batch_queue_pressure_ema
        ) + (0.2 * step_pressure)

    def _update_apply_commit_batch_queue_pressure_signals(self) -> None:
        current_total = self.apply_commit_batch_queue_evictions
        self.apply_commit_batch_queue_events_last_tick = max(
            0,
            current_total - self.apply_commit_batch_queue_events_prev_total,
        )
        self.apply_commit_batch_queue_events_prev_total = current_total
        self._update_apply_commit_batch_queue_pressure_ema()

    def _apply_commit_batch_queue_budget_backoff(self) -> int:
        limit = self._apply_commit_batch_queue_limit()
        if limit <= 1:
            return 0

        pressure = max(
            self._apply_commit_batch_queue_pressure_step(),
            self._apply_commit_batch_queue_pressure(),
            self.apply_commit_batch_queue_pressure_ema,
        )
        backoff = 0
        if pressure >= 1.0:
            backoff += 1
        if pressure >= 2.0:
            backoff += 1

        if self.cold_promotion_penalty >= 1.0 and backoff > 0:
            backoff -= 1
        if self.cold_promotion_penalty >= 1.5 and backoff > 0:
            backoff -= 1

        return min(max(0, backoff), max(0, limit - 1))

    def _adaptive_apply_commit_limit(self, *, background: bool) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 1
        limit = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        if self._apply_queue_pressure() >= 1.0:
            limit += 1
        if self.apply_queue_pressure_ema >= 1.0:
            limit += 1
        if self._apply_commit_queue_pressure() >= 1.0:
            limit += 1
        if self.apply_commit_queue_pressure_ema >= 1.0:
            limit += 1
        if self._apply_commit_batch_queue_pressure() >= 1.0:
            limit += 1
        if self.apply_commit_batch_queue_pressure_ema >= 1.0:
            limit += 1
        if self.cold_promotion_penalty >= 1.0:
            limit += 1
        if not background and self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        return min(max(1, limit), max(1, self._apply_commit_queue_limit()))

    def _adaptive_apply_commit_batch_limit(self, *, background: bool) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 1
        limit = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        if self._apply_commit_batch_queue_pressure() >= 1.0:
            limit += 1
        if self.apply_commit_batch_queue_pressure_ema >= 1.0:
            limit += 1
        if self.cold_promotion_penalty >= 1.0:
            limit += 1
        if self._apply_commit_batch_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_batch_queue_budget_backoff())
        if not background and self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        return min(max(1, limit), max(1, self._apply_commit_batch_queue_limit()))

    def _adaptive_activation_limit(self) -> int:
        base_limit = self._activated_cache_limit()
        if self.expert_prepared_cache_size is None:
            return base_limit
        controller_engaged = self._prepared_controller_engaged()
        effective_prepared_limit = max(1, int(self._effective_prepared_cache_limit() or base_limit))
        limit = base_limit
        if self._prepared_cache_pressure() >= 1.0 and self.prepared_cache_activation_stage_bonus <= 0.25:
            limit = max(1, base_limit - 1)
        if self.cold_promotion_penalty >= 1.0:
            limit = min(base_limit + 1, max(limit, base_limit))
        elif self.prepared_cache_activation_stage_bonus >= 1.0:
            limit = min(base_limit + 1, max(limit, base_limit))
        if controller_engaged:
            controller_cap = effective_prepared_limit
            if self.prepared_cache_activation_stage_bonus >= 1.0:
                controller_cap += 1
            if self.cold_promotion_penalty >= 1.0:
                controller_cap += 1
            limit = min(limit, max(1, controller_cap))
        if self.prepared_controller_aggressiveness >= 0.5:
            limit += 1
        if self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        if self._apply_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_queue_budget_backoff())
        if self._apply_commit_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_queue_budget_backoff())
        if self._apply_commit_batch_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_batch_queue_budget_backoff())
        return max(1, limit)

    def _adaptive_prebuild_limit(self) -> int:
        base_limit = max(1, self._activated_cache_limit() * 2)
        if self.expert_prepared_cache_size is None:
            return base_limit
        controller_engaged = self._prepared_controller_engaged()
        effective_prepared_limit = max(1, int(self._effective_prepared_cache_limit() or 1))
        pressure = self._prepared_cache_pressure()
        limit = base_limit
        if pressure >= 1.0:
            limit = max(1, min(base_limit, int(self.expert_prepared_cache_size)))
        if self.cold_promotion_penalty >= 1.0:
            limit = min(base_limit + 2, max(limit, int(self.expert_prepared_cache_size) + 2))
        elif self.prepared_cache_activation_stage_bonus >= 1.0:
            limit = min(base_limit + 1, max(limit, int(self.expert_prepared_cache_size) + 1))
        if controller_engaged:
            controller_cap = effective_prepared_limit + max(0, self._adaptive_activation_limit() - 1)
            if self.prepared_cache_activation_stage_bonus >= 1.0:
                controller_cap += 1
            if self.cold_promotion_penalty >= 1.0:
                controller_cap += 1
            limit = min(limit, max(1, controller_cap))
        if self.prepared_controller_aggressiveness >= 0.5:
            limit += 1
        if self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        if self._apply_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_queue_budget_backoff())
        if self._apply_commit_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_queue_budget_backoff())
        if self._apply_commit_batch_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_batch_queue_budget_backoff())
        return max(1, limit)

    def _adaptive_prefetch_pending_limit(self, *, phase: str) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 0
        base_limit = (
            max(1, int(self.dynamic_expert_scheduler.config.prefill_force_gpu_budget_per_layer))
            if phase == "prefill"
            else max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        )
        limit = max(base_limit, self._adaptive_activation_limit())
        if self.expert_prepared_cache_size is None:
            return limit

        if self._prepared_controller_engaged():
            limit = max(1, limit - self._prepared_cache_budget_backoff())
            if self._prepared_cache_rebalance_pressure_step() >= 1.0:
                limit = max(1, limit - 1)

        if self.cold_promotion_penalty >= 1.0:
            limit += 1
        if self.cold_promotion_penalty >= 1.5:
            limit += 1
        if self.prepared_controller_aggressiveness >= 0.5:
            limit += 1
        if self.prepared_controller_aggressiveness >= 1.0:
            limit += 1
        if self._apply_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_queue_budget_backoff())
        if self._apply_commit_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_queue_budget_backoff())
        if self._apply_commit_batch_queue_budget_backoff() > 0:
            limit = max(1, limit - self._apply_commit_batch_queue_budget_backoff())

        return min(max(1, limit), max(1, self._adaptive_prebuild_limit()))

    def _adaptive_prefetch_candidate_budget(self, *, phase: str) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 0
        base_budget = max(0, int(self.dynamic_expert_scheduler.config.prefetch_candidate_budget_per_layer))
        if phase == "prefill":
            base_budget = max(
                base_budget,
                int(self.dynamic_expert_scheduler.config.prefill_force_gpu_budget_per_layer),
            )
        if base_budget <= 0:
            return 0
        if self.expert_prepared_cache_size is None:
            return base_budget

        budget = base_budget
        if self._prepared_controller_engaged():
            budget = max(0, budget - self._prepared_cache_budget_backoff())
            if self._prepared_cache_rebalance_pressure_step() >= 1.0:
                budget = max(0, budget - 1)

        if self.cold_promotion_penalty >= 1.0:
            budget += 1
        if self.cold_promotion_penalty >= 1.5:
            budget += 1
        if self.prepared_controller_aggressiveness >= 0.5:
            budget += 1
        if self.prepared_controller_aggressiveness >= 1.0:
            budget += 1
        if self._apply_queue_budget_backoff() > 0:
            budget = max(0, budget - self._apply_queue_budget_backoff())
        if self._apply_commit_queue_budget_backoff() > 0:
            budget = max(0, budget - self._apply_commit_queue_budget_backoff())
        if self._apply_commit_batch_queue_budget_backoff() > 0:
            budget = max(0, budget - self._apply_commit_batch_queue_budget_backoff())

        return max(0, budget)

    def _update_cold_promotion_penalty(self, cold_promotions: int, total_promotions: int) -> None:
        if total_promotions <= 0:
            self.cold_promotion_penalty = max(0.0, self.cold_promotion_penalty - 0.1)
            return
        cold_ratio = cold_promotions / max(1, total_promotions)
        if cold_ratio >= 0.5:
            self.cold_promotion_penalty = min(2.0, self.cold_promotion_penalty + 0.5)
        elif cold_ratio == 0:
            self.cold_promotion_penalty = max(0.0, self.cold_promotion_penalty - 0.25)
        else:
            self.cold_promotion_penalty = max(0.0, self.cold_promotion_penalty - 0.1)

    def _activation_target_ids(self) -> set[int]:
        if self.offload_backend is None:
            return set()
        limit = self._adaptive_activation_limit()
        candidates: set[int] = {int(expert_idx) for expert_idx in self.activated_expert_cache.keys()}
        for op in self.offload_backend.migration_manager.peek_layer(self.layer_idx):
            if op.dst != ExpertResidency.GPU:
                continue
            state = self.offload_backend.migration_manager.state_for(self.layer_idx, int(op.expert_idx))
            if state not in {MigrationLifecycle.WARMED, MigrationLifecycle.ACTIVATED}:
                continue
            if bool(self.gpu_experts_mask[int(op.expert_idx)].item()):
                continue
            candidates.add(int(op.expert_idx))
        ordered = sorted(
            candidates,
            key=lambda expert_idx: (
                -self._migration_state_priority(expert_idx),
                -self._hotness_score(expert_idx),
                int(expert_idx),
            ),
        )
        return set(ordered[:limit])

    def _prebuild_target_ids(self) -> set[int]:
        if self.offload_backend is None:
            return set()
        limit = self._adaptive_prebuild_limit()
        candidates: set[int] = set()
        for op in self.offload_backend.migration_manager.peek_layer(self.layer_idx):
            if op.dst != ExpertResidency.GPU:
                continue
            expert_idx = int(op.expert_idx)
            if bool(self.gpu_experts_mask[expert_idx].item()):
                continue
            state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if state not in {MigrationLifecycle.READY, MigrationLifecycle.WARMED, MigrationLifecycle.ACTIVATED}:
                continue
            candidates.add(expert_idx)
        ordered = sorted(
            candidates,
            key=lambda expert_idx: (
                -self._migration_state_priority(expert_idx),
                -self._hotness_score(expert_idx),
                int(expert_idx),
            ),
        )
        return set(ordered[:limit])

    def _store_activated_module(self, expert_idx: int, module: nn.Module) -> None:
        expert_key = str(expert_idx)
        self.activated_expert_cache[expert_key] = module
        self.activated_expert_cache.move_to_end(expert_key)
        self.activated_cache_stores += 1
        while len(self.activated_expert_cache) > self._activated_cache_limit():
            evicted_key = self._pick_activated_cache_victim_key()
            if evicted_key is None:
                evicted_key, evicted_module = self.activated_expert_cache.popitem(last=False)
            else:
                evicted_module = self.activated_expert_cache.pop(evicted_key)
            self._insert_warm_module(int(evicted_key), evicted_module)
            if self.offload_backend is not None:
                evicted_idx = int(evicted_key)
                state = self.offload_backend.migration_manager.state_for(self.layer_idx, evicted_idx)
                if state == MigrationLifecycle.ACTIVATED:
                    self.offload_backend.migration_manager.mark_state(
                        self.layer_idx,
                        evicted_idx,
                        state=MigrationLifecycle.WARMED,
                    )
            self.activated_cache_evictions += 1
        self._rebalance_prepared_caches()

    def _activate_warm_module(self, module: nn.Module, device: torch.device, dtype: torch.dtype) -> nn.Module:
        non_blocking = device.type == "cuda"
        self.warm_cache_device_transfers += 1
        return module.to(device=device, dtype=dtype, non_blocking=non_blocking)

    def _activate_warmed_experts(
        self,
        *,
        phase: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        if phase != "decode" or self.offload_backend is None:
            return 0

        target_ids = self._activation_target_ids()
        for expert_key in list(self.activated_expert_cache.keys()):
            expert_idx = int(expert_key)
            if expert_idx in target_ids:
                continue
            module = self.activated_expert_cache.pop(expert_key)
            self._store_warm_module(expert_idx, module, count_store=False)
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                expert_idx,
                phase=phase,
                state=MigrationLifecycle.WARMED,
            )
            self.activated_cache_evictions += 1

        activated = 0
        for op in self.offload_backend.migration_manager.peek_layer(self.layer_idx):
            if op.dst != ExpertResidency.GPU:
                continue
            expert_idx = int(op.expert_idx)
            expert_key = str(expert_idx)
            current_state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if current_state != MigrationLifecycle.WARMED:
                continue
            if expert_idx not in target_ids:
                continue
            if (
                expert_key in self.gpu_experts
                or expert_key in self.activated_expert_cache
                or expert_key not in self.warm_expert_cache
            ):
                continue
            if bool(self.gpu_experts_mask[expert_idx].item()):
                continue

            self.activation_submitted += 1
            warm_module = self.warm_expert_cache.pop(expert_key)
            activated_module = self._activate_warm_module(warm_module, device, dtype)
            self._store_activated_module(expert_idx, activated_module)
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                expert_idx,
                src=op.src.value,
                dst=op.dst.value,
                reason=op.reason,
                phase=phase,
                state=MigrationLifecycle.ACTIVATED,
            )
            activated += 1

        self.activation_ready += activated
        return activated

    def _enqueue_activated_apply_candidates(self, *, phase: str) -> int:
        if phase != "decode" or self.offload_backend is None:
            return 0

        enqueued = 0
        for op in self._coalesce_migration_ops(self.offload_backend.migration_manager.peek_layer(self.layer_idx)):
            if op.dst != ExpertResidency.GPU:
                continue
            expert_idx = int(op.expert_idx)
            expert_key = str(expert_idx)
            if bool(self.gpu_experts_mask[expert_idx].item()):
                self.apply_candidate_queue.pop(expert_key, None)
                continue
            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if lifecycle != MigrationLifecycle.ACTIVATED:
                continue
            was_present = expert_key in self.apply_candidate_queue
            self.apply_candidate_queue[expert_key] = op
            self.apply_candidate_queue.move_to_end(expert_key)
            if not was_present:
                self.apply_queue_enqueued += 1
                enqueued += 1
        self._rebalance_apply_candidate_queue()
        return enqueued

    def _enqueue_apply_commit_candidates(
        self,
        *,
        expert_ids: set[int],
        background: bool,
    ) -> int:
        enqueued = 0
        for expert_idx in expert_ids:
            expert_key = str(int(expert_idx))
            op = self.apply_candidate_queue.get(expert_key)
            if op is None:
                continue
            was_present = expert_key in self.apply_commit_queue
            self.apply_commit_queue[expert_key] = op
            self.apply_commit_queue.move_to_end(expert_key)
            if not was_present:
                self.apply_commit_queue_enqueued += 1
                enqueued += 1
                if background:
                    self.background_apply_commit_queue_enqueued += 1
        self._rebalance_apply_commit_queue()
        return enqueued

    def _stage_apply_commit_batch(
        self,
        *,
        phase: str,
        active_experts: Optional[set[int]] = None,
        eligible_expert_ids: Optional[set[int]] = None,
        max_commits: Optional[int] = None,
        background: bool = False,
    ) -> int:
        if phase != "decode" or self.offload_backend is None:
            return 0

        active_experts = active_experts or set()
        candidate_ops = []
        for expert_key, op in self.apply_candidate_queue.items():
            expert_idx = int(expert_key)
            if eligible_expert_ids is not None and expert_idx not in eligible_expert_ids:
                continue
            if bool(self.gpu_experts_mask[expert_idx].item()):
                continue
            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if lifecycle != MigrationLifecycle.ACTIVATED:
                continue
            candidate_ops.append(op)
        if not candidate_ops:
            return 0

        candidate_ops.sort(key=lambda op: self._promotion_sort_key(op, active_experts, phase))
        if max_commits is None:
            max_commits = self._adaptive_apply_commit_limit(background=background)
        else:
            max_commits = max(1, int(max_commits))
        selected_ops, _deferred = self._select_ready_promotion_batch(
            candidate_ops,
            max_promotions=max_commits,
        )
        selected_ids = {int(op.expert_idx) for op in selected_ops}
        if not selected_ids:
            return 0
        return self._enqueue_apply_commit_candidates(
            expert_ids=selected_ids,
            background=background,
        )

    def _prune_apply_candidate_queue(self) -> int:
        stale_keys: list[str] = []
        for expert_key in list(self.apply_candidate_queue.keys()):
            expert_idx = int(expert_key)
            lifecycle = None
            if self.offload_backend is not None:
                lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if bool(self.gpu_experts_mask[expert_idx].item()) or lifecycle != MigrationLifecycle.ACTIVATED:
                stale_keys.append(expert_key)

        for expert_key in stale_keys:
            self.apply_candidate_queue.pop(expert_key, None)
        self.apply_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_apply_commit_queue(self) -> int:
        stale_keys: list[str] = []
        for expert_key in list(self.apply_commit_queue.keys()):
            expert_idx = int(expert_key)
            lifecycle = None
            if self.offload_backend is not None:
                lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if (
                bool(self.gpu_experts_mask[expert_idx].item())
                or lifecycle not in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                or expert_key not in self.apply_candidate_queue
            ):
                stale_keys.append(expert_key)

        for expert_key in stale_keys:
            self.apply_commit_queue.pop(expert_key, None)
        self.apply_commit_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_apply_commit_batch_queue(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.apply_commit_batch_queue.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                expert_key = str(expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if (
                    expert_key in self.apply_commit_queue
                    and expert_key in self.apply_commit_ready_cache
                    and not bool(self.gpu_experts_mask[expert_idx].item())
                    and lifecycle in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                ):
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.apply_commit_batch_queue[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.apply_commit_batch_queue.pop(batch_key, None)
        self.apply_commit_batch_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_resident_commit_batch_queue(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.resident_commit_batch_queue.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                expert_key = str(expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if (
                    expert_key in self.apply_commit_ready_cache
                    and not bool(self.gpu_experts_mask[expert_idx].item())
                    and lifecycle in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                ):
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.resident_commit_batch_queue[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.resident_commit_batch_queue.pop(batch_key, None)
        self.resident_commit_batch_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_resident_commit_finalize_queue(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.resident_commit_finalize_queue.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                expert_key = str(expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if (
                    expert_key in self.apply_commit_ready_cache
                    and not bool(self.gpu_experts_mask[expert_idx].item())
                    and lifecycle in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                ):
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.resident_commit_finalize_queue[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.resident_commit_finalize_queue.pop(batch_key, None)
        self.resident_commit_finalize_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_resident_commit_ready_cache(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.resident_commit_ready_cache.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                expert_key = str(expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if (
                    expert_key in self.apply_commit_ready_cache
                    and lifecycle in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                ):
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.resident_commit_ready_cache[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.resident_commit_ready_cache.pop(batch_key, None)
        self.resident_commit_ready_cache_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_resident_commit_apply_queue(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.resident_commit_apply_queue.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if bool(self.gpu_experts_mask[expert_idx].item()) or lifecycle == MigrationLifecycle.APPLIED:
                    continue
                if str(expert_idx) in self.apply_commit_ready_cache and lifecycle in {
                    MigrationLifecycle.ACTIVATED,
                    MigrationLifecycle.APPLIED,
                }:
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.resident_commit_apply_queue[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.resident_commit_apply_queue.pop(batch_key, None)
        self.resident_commit_apply_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _prune_resident_commit_finalize_ready_queue(self) -> int:
        stale_keys: list[str] = []
        for batch_key, batch_entries in list(self.resident_commit_finalize_ready_queue.items()):
            retained_entries: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                lifecycle = None
                if self.offload_backend is not None:
                    lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if bool(self.gpu_experts_mask[expert_idx].item()) or lifecycle == MigrationLifecycle.APPLIED:
                    continue
                if (
                    batch_key in self.resident_commit_apply_queue
                    and str(expert_idx) in self.apply_commit_ready_cache
                    and lifecycle in {MigrationLifecycle.ACTIVATED, MigrationLifecycle.APPLIED}
                ):
                    retained_entries.append((op, resolved))
            if retained_entries:
                self.resident_commit_finalize_ready_queue[batch_key] = retained_entries
            else:
                stale_keys.append(batch_key)

        for batch_key in stale_keys:
            self.resident_commit_finalize_ready_queue.pop(batch_key, None)
        self.resident_commit_finalize_ready_queue_pruned += len(stale_keys)
        return len(stale_keys)

    def _stage_resident_commit_ready_cache(
        self,
        *,
        eligible_batch_keys: Optional[set[str]] = None,
        max_batches: Optional[int],
        background: bool,
    ) -> int:
        if not self.resident_commit_finalize_queue:
            return 0

        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batches is None
            else max(1, int(max_batches))
        )
        candidate_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in list(self.resident_commit_finalize_queue.items()):
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            if not batch_entries:
                continue
            candidate_batches.append((batch_key, batch_entries))

        if not candidate_batches:
            return 0

        candidate_batches.sort(
            key=lambda item: min(
                self._promotion_sort_key(op, set(), "decode")
                for op, _resolved in item[1]
            )
        )
        selected_batches = candidate_batches[:batch_limit]
        enqueued = 0
        for batch_key, batch_entries in selected_batches:
            was_present = batch_key in self.resident_commit_ready_cache
            self.resident_commit_ready_cache[batch_key] = batch_entries
            self.resident_commit_ready_cache.move_to_end(batch_key)
            if not was_present:
                self.resident_commit_ready_cache_stores += 1
                enqueued += 1
                if background:
                    self.background_resident_commit_ready_cache_stores += 1

        self._rebalance_resident_commit_ready_cache()
        return enqueued

    def _stage_resident_commit_apply_queue(
        self,
        *,
        eligible_batch_keys: Optional[set[str]] = None,
        max_batches: Optional[int],
        background: bool,
    ) -> int:
        if not self.resident_commit_ready_cache:
            return 0

        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batches is None
            else max(1, int(max_batches))
        )
        candidate_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in list(self.resident_commit_ready_cache.items()):
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            if not batch_entries:
                continue
            candidate_batches.append((batch_key, batch_entries))

        if not candidate_batches:
            return 0

        candidate_batches.sort(
            key=lambda item: min(
                self._promotion_sort_key(op, set(), "decode")
                for op, _resolved in item[1]
            )
        )
        selected_batches = candidate_batches[:batch_limit]
        enqueued = 0
        for batch_key, batch_entries in selected_batches:
            was_present = batch_key in self.resident_commit_apply_queue
            self.resident_commit_apply_queue[batch_key] = batch_entries
            self.resident_commit_apply_queue.move_to_end(batch_key)
            if not was_present:
                self.resident_commit_apply_queue_enqueued += 1
                self.resident_commit_apply_queue_batches += 1
                enqueued += 1
                if background:
                    self.background_resident_commit_apply_queue_enqueued += 1

        self._rebalance_resident_commit_apply_queue()
        return enqueued

    def _stage_resident_commit_finalize_ready_queue(
        self,
        *,
        eligible_batch_keys: Optional[set[str]] = None,
        max_batches: Optional[int],
        background: bool,
    ) -> int:
        if not self.resident_commit_apply_queue:
            return 0

        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batches is None
            else max(1, int(max_batches))
        )
        candidate_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in list(self.resident_commit_apply_queue.items()):
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            if not batch_entries:
                continue
            candidate_batches.append((batch_key, batch_entries))

        if not candidate_batches:
            return 0

        candidate_batches.sort(
            key=lambda item: min(
                self._promotion_sort_key(op, set(), "decode")
                for op, _resolved in item[1]
            )
        )
        selected_batches = candidate_batches[:batch_limit]
        enqueued = 0
        for batch_key, batch_entries in selected_batches:
            was_present = batch_key in self.resident_commit_finalize_ready_queue
            self.resident_commit_finalize_ready_queue[batch_key] = batch_entries
            self.resident_commit_finalize_ready_queue.move_to_end(batch_key)
            if not was_present:
                self.resident_commit_finalize_ready_queue_enqueued += 1
                self.resident_commit_finalize_ready_queue_batches += 1
                enqueued += 1
                if background:
                    self.background_resident_commit_finalize_ready_queue_enqueued += 1

        self._rebalance_resident_commit_finalize_ready_queue()
        return enqueued

    def _stage_resident_commit_finalize_queue(
        self,
        *,
        eligible_batch_keys: Optional[set[str]] = None,
        max_batches: Optional[int],
        background: bool,
    ) -> int:
        if not self.resident_commit_batch_queue:
            return 0

        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batches is None
            else max(1, int(max_batches))
        )
        candidate_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in list(self.resident_commit_batch_queue.items()):
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            if not batch_entries:
                continue
            candidate_batches.append((batch_key, batch_entries))

        if not candidate_batches:
            return 0

        candidate_batches.sort(
            key=lambda item: min(
                self._promotion_sort_key(op, set(), "decode")
                for op, _resolved in item[1]
            )
        )
        selected_batches = candidate_batches[:batch_limit]
        enqueued = 0
        for batch_key, batch_entries in selected_batches:
            was_present = batch_key in self.resident_commit_finalize_queue
            self.resident_commit_finalize_queue[batch_key] = batch_entries
            self.resident_commit_finalize_queue.move_to_end(batch_key)
            if not was_present:
                self.resident_commit_finalize_queue_enqueued += 1
                self.resident_commit_finalize_queue_batches += 1
                enqueued += 1
                if background:
                    self.background_resident_commit_finalize_queue_enqueued += 1

        self._rebalance_resident_commit_finalize_queue()
        return enqueued

    def _commit_apply_candidate_queue(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        phase: str,
        active_experts: Optional[set[int]] = None,
        eligible_expert_ids: Optional[set[int]] = None,
        eligible_batch_keys: Optional[set[str]] = None,
        max_commits: Optional[int] = None,
        max_batch_commits: Optional[int] = None,
        allow_eviction: bool,
        count_batch: bool,
        background: bool = False,
        stage_resident_batches: bool = True,
    ) -> tuple[int, int]:
        if (
            phase != "decode"
            or self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
        ):
            return 0, 0

        self._prune_apply_candidate_queue()
        self._prune_apply_commit_queue()
        self._prune_apply_commit_batch_queue()
        self._prune_resident_commit_batch_queue()
        self._prune_resident_commit_finalize_queue()
        self._prune_resident_commit_ready_cache()
        self._prune_resident_commit_apply_queue()
        self._prune_resident_commit_finalize_ready_queue()
        self._prune_apply_commit_ready_cache()
        if not self.apply_commit_queue:
            return 0, 0

        active_experts = active_experts or set()
        stage_limit = (
            max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
            if max_commits is None
            else max(1, int(max_commits))
        )
        batch_limit = (
            self._adaptive_apply_commit_batch_limit(background=background)
            if max_batch_commits is None and max_commits is None
            else max(1, int(max_batch_commits if max_batch_commits is not None else max_commits))
        )
        self._resolve_apply_commit_queue(
            device=device,
            dtype=dtype,
            eligible_expert_ids=eligible_expert_ids,
            max_resolves=stage_limit,
            background=background,
        )
        self._stage_apply_commit_batch_queue(
            eligible_expert_ids=eligible_expert_ids,
            max_commits=batch_limit,
            background=background,
        )
        if stage_resident_batches:
            self._stage_resident_commit_batches(
                eligible_batch_keys=eligible_batch_keys,
                max_batches=batch_limit,
                background=background,
            )
        self._stage_resident_commit_finalize_queue(
            eligible_batch_keys=eligible_batch_keys,
            max_batches=batch_limit,
            background=background,
        )
        self._stage_resident_commit_ready_cache(
            eligible_batch_keys=eligible_batch_keys,
            max_batches=batch_limit,
            background=background,
        )
        self._stage_resident_commit_apply_queue(
            eligible_batch_keys=eligible_batch_keys,
            max_batches=batch_limit,
            background=background,
        )
        self._stage_resident_commit_finalize_ready_queue(
            eligible_batch_keys=eligible_batch_keys,
            max_batches=batch_limit,
            background=background,
        )
        ready_batches: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        for batch_key, batch_entries in self.resident_commit_finalize_ready_queue.items():
            if eligible_batch_keys is not None and batch_key not in eligible_batch_keys:
                continue
            filtered_batch: list[tuple[object, dict[str, object]]] = []
            for op, resolved in batch_entries:
                expert_idx = int(op.expert_idx)
                if eligible_expert_ids is not None and expert_idx not in eligible_expert_ids:
                    continue
                if str(expert_idx) not in self.apply_commit_ready_cache:
                    continue
                filtered_batch.append((op, resolved))
            if filtered_batch:
                ready_batches.append((batch_key, filtered_batch))
        if not ready_batches:
            return 0, 0

        selected_batch_entries = ready_batches[:batch_limit]
        selected_ops = [
            op
            for _batch_key, batch_entries in selected_batch_entries
            for op, _resolved in batch_entries
        ]
        deferred = max(
            0,
            sum(len(batch_entries) for _batch_key, batch_entries in ready_batches[batch_limit:]),
        )

        if not selected_ops:
            return 0, deferred

        gpu_budget = self._runtime_gpu_budget()
        current_gpu_residents = int(self.gpu_experts_mask.bool().sum().item())
        free_slots = max(0, gpu_budget - current_gpu_residents)

        selected_expert_count = sum(len(batch_entries) for _batch_key, batch_entries in selected_batch_entries)

        if allow_eviction and gpu_budget > 0 and selected_expert_count > free_slots:
            protected_experts = set(active_experts)
            protected_experts.update(int(op.expert_idx) for op in selected_ops)
            required_slots = max(0, selected_expert_count - free_slots)
            evicted = self._evict_for_promotion_batch(
                protected_experts=protected_experts,
                fallback_dst=self.dynamic_expert_scheduler.config.offload_tier,
                required_slots=required_slots,
                phase=phase,
            )
            promotable_slots = free_slots + evicted
        else:
            promotable_slots = free_slots

        promotable_batch_entries: list[tuple[str, list[tuple[object, dict[str, object]]]]] = []
        committed_experts = 0
        for batch_key, batch_entries in selected_batch_entries:
            batch_size = len(batch_entries)
            if batch_size <= 0:
                continue
            if committed_experts + batch_size > promotable_slots:
                deferred += batch_size
                continue
            promotable_batch_entries.append((batch_key, batch_entries))
            committed_experts += batch_size

        if not promotable_batch_entries:
            return 0, deferred

        ready_entries = [
            (op, resolved)
            for _batch_key, batch_entries in promotable_batch_entries
            for op, resolved in batch_entries
        ]
        if not ready_entries:
            return 0, deferred

        applied, completed_expert_ids, _source_counts = self._apply_promotion_batch(
            [op for op, _resolved in ready_entries],
            device=device,
            dtype=dtype,
            phase=phase,
            pre_resolved_batch=ready_entries,
        )
        if completed_expert_ids:
            for expert_idx in completed_expert_ids:
                self.apply_candidate_queue.pop(str(expert_idx), None)
                self.apply_commit_queue.pop(str(expert_idx), None)
                self.apply_commit_ready_cache.pop(str(expert_idx), None)
            for batch_key, _batch_entries in promotable_batch_entries:
                self.apply_commit_batch_queue.pop(batch_key, None)
                self.resident_commit_batch_queue.pop(batch_key, None)
                self.resident_commit_finalize_queue.pop(batch_key, None)
                self.resident_commit_ready_cache.pop(batch_key, None)
                self.resident_commit_apply_queue.pop(batch_key, None)
                self.resident_commit_finalize_ready_queue.pop(batch_key, None)
            self.offload_backend.migration_manager.take_layer(
                self.layer_idx,
                lambda op: (
                    op.dst == ExpertResidency.GPU
                    and int(op.expert_idx) in completed_expert_ids
                ),
            )
        self.apply_queue_committed += applied
        if applied > 0:
            self.apply_queue_commit_batches += 1
            self.apply_queue_commit_experts += applied
            self.apply_commit_batch_queue_committed_batches += len(promotable_batch_entries)
            self.resident_commit_batch_queue_committed_batches += len(promotable_batch_entries)
            self.resident_commit_finalize_queue_committed_batches += len(promotable_batch_entries)
            self.resident_commit_apply_queue_committed_batches += len(promotable_batch_entries)
            self.resident_commit_finalize_ready_queue_committed_batches += len(promotable_batch_entries)
            if background:
                self.background_apply_commit_batches += 1
                self.background_apply_commit_experts += applied
                self.background_apply_commit_batch_queue_committed_batches += len(promotable_batch_entries)
                self.background_resident_commit_batch_queue_committed_batches += len(promotable_batch_entries)
                self.background_resident_commit_finalize_queue_committed_batches += len(promotable_batch_entries)
                self.background_resident_commit_apply_queue_committed_batches += len(promotable_batch_entries)
                self.background_resident_commit_finalize_ready_queue_committed_batches += len(promotable_batch_entries)
        return applied, deferred

    def _background_apply_activated_experts(
        self,
        *,
        phase: str,
        eligible_expert_ids: Optional[set[int]] = None,
        eligible_batch_keys: Optional[set[str]] = None,
        stage_resident_batches: bool = True,
    ) -> int:
        if (
            phase != "decode"
            or self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
        ):
            return 0

        applied, _deferred = self._commit_apply_candidate_queue(
            device=torch.device("cpu"),
            dtype=torch.float32,
            phase=phase,
            eligible_expert_ids=eligible_expert_ids,
            eligible_batch_keys=eligible_batch_keys,
            max_commits=self._adaptive_apply_commit_limit(background=True),
            max_batch_commits=self._adaptive_apply_commit_batch_limit(background=True),
            allow_eviction=False,
            count_batch=False,
            background=True,
            stage_resident_batches=stage_resident_batches,
        )
        if applied:
            self.background_activation_applied += applied
            self._synchronize_gpu_mask()
        return applied

    def _prebuild_ready_experts(self, *, phase: str, device: torch.device, dtype: torch.dtype) -> int:
        if (
            phase != "decode"
            or self.offload_backend is None
            or self.expert_warm_cache_size <= 0
        ):
            return 0
        target_ids = self._prebuild_target_ids()
        built = 0
        for op in self.offload_backend.migration_manager.peek_layer(self.layer_idx):
            if op.dst != ExpertResidency.GPU:
                continue
            expert_idx = int(op.expert_idx)
            if expert_idx not in target_ids:
                continue
            current_state = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            if current_state not in {
                MigrationLifecycle.READY,
                MigrationLifecycle.WARMED,
                MigrationLifecycle.ACTIVATED,
            }:
                continue
            expert_key = str(expert_idx)
            if (
                expert_key in self.gpu_experts
                or expert_key in self.warm_expert_cache
                or expert_key in self.activated_expert_cache
            ):
                continue
            module = self._build_runtime_expert(expert_idx, torch.device("cpu"), dtype)
            self._store_warm_module(expert_idx, module, count_store=False)
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                expert_idx,
                src=op.src.value,
                dst=op.dst.value,
                reason=op.reason,
                phase=phase,
                state=MigrationLifecycle.WARMED,
            )
            built += 1
        self.warm_cache_prebuilt += built
        return built

    def _promote_ready_migrations(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        phase: str,
    ) -> tuple[int, int, int]:
        if (
            self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
            or phase != "decode"
        ):
            return 0, 0, 0

        active_experts: set[int] = set()
        apply_queue_enqueued = self._enqueue_activated_apply_candidates(phase=phase)
        apply_commit_queue_enqueued = self._stage_apply_commit_batch(
            phase=phase,
            active_experts=active_experts,
        )
        max_promotions = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        applied_from_queue, deferred_from_queue = self._commit_apply_candidate_queue(
            device=device,
            dtype=dtype,
            phase=phase,
            max_commits=self._adaptive_apply_commit_limit(background=False),
            max_batch_commits=self._adaptive_apply_commit_batch_limit(background=False),
            allow_eviction=True,
            count_batch=True,
            background=False,
        )
        remaining_promotions = max(0, max_promotions - applied_from_queue)
        queued_ops = self.offload_backend.migration_manager.peek_layer(self.layer_idx)
        if not queued_ops:
            return applied_from_queue, deferred_from_queue, apply_queue_enqueued + apply_commit_queue_enqueued
        if remaining_promotions <= 0:
            deferred_ready = 0
            for op in self._coalesce_migration_ops(queued_ops):
                if op.dst != ExpertResidency.GPU:
                    continue
                expert_idx = int(op.expert_idx)
                if bool(self.gpu_experts_mask[expert_idx].item()):
                    continue
                lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
                if lifecycle in {MigrationLifecycle.READY, MigrationLifecycle.WARMED}:
                    deferred_ready += 1
            return applied_from_queue, deferred_from_queue + deferred_ready, apply_queue_enqueued + apply_commit_queue_enqueued

        queued_ops = self._coalesce_migration_ops(queued_ops)
        promotion_ops = []
        for op in queued_ops:
            if op.dst != ExpertResidency.GPU:
                continue
            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, int(op.expert_idx))
            if lifecycle == MigrationLifecycle.ACTIVATED:
                continue
            if lifecycle not in {
                MigrationLifecycle.READY,
                MigrationLifecycle.WARMED,
            }:
                continue
            promotion_ops.append(op)
        if not promotion_ops:
            return applied_from_queue, deferred_from_queue, apply_queue_enqueued + apply_commit_queue_enqueued

        promotion_ops.sort(key=lambda op: self._promotion_sort_key(op, set(), phase))
        promotion_ops, deferred = self._select_ready_promotion_batch(
            promotion_ops,
            max_promotions=remaining_promotions,
        )
        if not promotion_ops:
            return applied_from_queue, deferred + deferred_from_queue, apply_queue_enqueued + apply_commit_queue_enqueued
        gpu_budget = self._runtime_gpu_budget()
        protected_experts = {int(op.expert_idx) for op in promotion_ops}
        resident_ops = [op for op in promotion_ops if bool(self.gpu_experts_mask[int(op.expert_idx)].item())]
        pending_ops = [op for op in promotion_ops if not bool(self.gpu_experts_mask[int(op.expert_idx)].item())]

        applied = applied_from_queue
        completed_expert_ids: set[int] = set()
        for op in resident_ops:
            expert_idx = int(op.expert_idx)
            if self.offload_backend is not None:
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.APPLIED,
                )
            completed_expert_ids.add(expert_idx)

        if gpu_budget > 0 and pending_ops:
            current_gpu_residents = int(self.gpu_experts_mask.bool().sum().item())
            free_slots = max(0, gpu_budget - current_gpu_residents)
            required_slots = max(0, len(pending_ops) - free_slots)
            evicted = self._evict_for_promotion_batch(
                protected_experts=protected_experts,
                fallback_dst=self.dynamic_expert_scheduler.config.offload_tier,
                required_slots=required_slots,
                phase=phase,
            )
            promotable = min(len(pending_ops), free_slots + evicted)
            deferred += max(0, len(pending_ops) - promotable)
            pending_ops = pending_ops[:promotable]

        if pending_ops:
            batch_applied, batch_completed, _source_counts = self._apply_promotion_batch(
                pending_ops,
                device=device,
                dtype=dtype,
                phase=phase,
            )
            applied += batch_applied
            completed_expert_ids.update(batch_completed)

        if applied:
            self.last_applied_migration_phase = phase
            self._synchronize_gpu_mask()
        if completed_expert_ids:
            self.offload_backend.migration_manager.take_layer(
                    self.layer_idx,
                    lambda op: (
                        op.dst == ExpertResidency.GPU
                        and int(op.expert_idx) in completed_expert_ids
                    ),
                )
        self.applied_migration_history = self.applied_migration_history[-64:]
        return applied, deferred + deferred_from_queue, apply_queue_enqueued + apply_commit_queue_enqueued

    def advance_offload_pipeline(
        self,
        *,
        phase: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, int]:
        with self._pipeline_lock:
            prev_apply_batches = self.pipeline_apply_batches
            prev_apply_batch_experts = self.pipeline_apply_batch_experts
            prev_apply_batch_evictions = self.pipeline_apply_batch_evictions
            prev_apply_batch_activated = self.pipeline_apply_batch_activated
            prev_apply_batch_warm = self.pipeline_apply_batch_warm
            prev_apply_batch_cold = self.pipeline_apply_batch_cold
            prefetch_submitted = self._prime_pending_promotions(phase=phase)
            ready_polled = self.refresh_offload_state()
            warm_prebuilt = self._prebuild_ready_experts(phase=phase, device=device, dtype=dtype)
            activation_ready = self._activate_warmed_experts(phase=phase, device=device, dtype=dtype)
            ready_applied = 0
            ready_deferred = 0
            apply_queue_enqueued = 0
            if phase == "decode":
                ready_applied, ready_deferred, apply_queue_enqueued = self._promote_ready_migrations(
                    device=device,
                    dtype=dtype,
                    phase=phase,
                )
            self.pipeline_ticks += 1
            self._update_prepared_cache_rebalance_pressure_signals()
            self._update_apply_queue_pressure_signals()
            self._update_apply_commit_queue_pressure_signals()
            self._update_apply_commit_batch_queue_pressure_signals()
            self.pipeline_ready_applied += ready_applied
            self.pipeline_ready_deferred += ready_deferred
            return {
                "ready_polled": int(ready_polled),
                "ready_applied": int(ready_applied),
                "ready_deferred": int(ready_deferred),
                "prefetch_submitted": int(prefetch_submitted),
                "apply_queue_enqueued": int(apply_queue_enqueued),
                "warm_prebuilt": int(warm_prebuilt),
                "activation_ready": int(activation_ready),
                "apply_batch_count": int(self.pipeline_apply_batches - prev_apply_batches),
                "apply_batch_experts": int(self.pipeline_apply_batch_experts - prev_apply_batch_experts),
                "apply_batch_evictions": int(self.pipeline_apply_batch_evictions - prev_apply_batch_evictions),
                "apply_batch_activated": int(self.pipeline_apply_batch_activated - prev_apply_batch_activated),
                "apply_batch_warm": int(self.pipeline_apply_batch_warm - prev_apply_batch_warm),
                "apply_batch_cold": int(self.pipeline_apply_batch_cold - prev_apply_batch_cold),
            }

    def _request_prefetch_candidates(self, *, phase: str) -> None:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return
        candidates = self.dynamic_expert_scheduler.prefetch_candidates_layer(self.layer_idx, phase=phase)
        if not candidates:
            return
        candidate_budget = self._adaptive_prefetch_candidate_budget(phase=phase)
        if candidate_budget <= 0:
            return
        self.prefetch_candidate_scans += 1
        submitted = 0
        for expert_idx in candidates:
            if not self.materialization_manager.has_cached(self.layer_idx, int(expert_idx)):
                before = self.prefetch_enqueued
                self._request_prefetch(int(expert_idx))
                if self.prefetch_enqueued > before:
                    submitted += 1
                if submitted >= candidate_budget:
                    break

    def _set_residency(self, expert_idx: int, residency: ExpertResidency) -> None:
        if self.residency_plan is None:
            return
        state = self.residency_plan.layer_state(self.layer_idx)
        step = state.logical_step
        state.record_residency_change(expert_idx, residency, step=step)

    def _runtime_gpu_budget(self) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return int(self.gpu_experts_mask.bool().sum().item())
        return max(0, int(self.dynamic_expert_scheduler.config.gpu_budget_per_layer))

    def _runtime_gpu_residents(self) -> list[int]:
        return [int(expert_idx) for expert_idx in torch.where(self.gpu_experts_mask.bool())[0].tolist()]

    def _synchronize_gpu_mask(self) -> None:
        if self.residency_plan is not None:
            layer_state = self.residency_plan.layer_state(self.layer_idx)
            self.gpu_experts_mask = layer_state.gpu_mask().to(device=self.gpu_experts_mask.device)
        if self.offload_backend is not None:
            self.offload_backend.update_gpu_expert_mask(self.gpu_experts_mask)

    def _promote_expert_to_gpu(self, expert_idx: int, device: torch.device, dtype: torch.dtype) -> str | None:
        expert_key = str(expert_idx)
        source = "cold"
        if expert_key not in self.gpu_experts:
            resolved = self._resolve_promotion_module(expert_idx, device, dtype)
            self.gpu_experts[expert_key] = resolved["module"]
            source = str(resolved["source"])
        self.activation_applied += 1
        self.gpu_experts_mask[expert_idx] = True
        self._set_residency(expert_idx, ExpertResidency.GPU)
        return source

    def _resolve_promotion_module(
        self,
        expert_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        expert_key = str(expert_idx)
        if expert_key in self.gpu_experts:
            return {
                "expert_idx": int(expert_idx),
                "module": self.gpu_experts[expert_key],
                "source": "resident",
            }

        activated_module = self.activated_expert_cache.pop(expert_key, None)
        if activated_module is not None:
            self.activated_cache_hits += 1
            return {
                "expert_idx": int(expert_idx),
                "module": activated_module.to(dtype=dtype),
                "source": "activated",
            }

        warm_module = self.warm_expert_cache.pop(expert_key, None)
        if warm_module is not None:
            self.warm_cache_hits += 1
            module_device = next(iter(warm_module.parameters())).device
            if module_device != device:
                warm_module = self._activate_warm_module(warm_module, device, dtype)
            else:
                warm_module = warm_module.to(dtype=dtype)
            return {
                "expert_idx": int(expert_idx),
                "module": warm_module,
                "source": "warm",
            }

        return {
            "expert_idx": int(expert_idx),
            "module": self._build_runtime_expert(expert_idx, device, dtype),
            "source": "cold",
        }

    def _apply_promotion_batch(
        self,
        promotion_ops: list,
        *,
        device: torch.device,
        dtype: torch.dtype,
        phase: str,
        pre_resolved_batch: Optional[list[tuple[object, dict[str, object]]]] = None,
    ) -> tuple[int, set[int], dict[str, int]]:
        applied = 0
        completed_expert_ids: set[int] = set()
        source_counts = {"activated": 0, "warm": 0, "cold": 0}
        resolved_batch = (
            pre_resolved_batch
            if pre_resolved_batch is not None
            else [
                (
                    op,
                    self._resolve_promotion_module(int(op.expert_idx), device, dtype),
                )
                for op in promotion_ops
            ]
        )

        modules_to_commit: "OrderedDict[str, nn.Module]" = OrderedDict()
        batch_expert_indices: list[int] = []
        for op, resolved in resolved_batch:
            expert_idx = int(op.expert_idx)
            expert_key = str(expert_idx)
            if expert_key not in self.gpu_experts and expert_key not in modules_to_commit:
                modules_to_commit[expert_key] = resolved["module"]  # type: ignore[index]
            batch_expert_indices.append(expert_idx)

        if modules_to_commit:
            self.gpu_experts.update(modules_to_commit)
        if batch_expert_indices:
            expert_indices = torch.tensor(
                batch_expert_indices,
                device=self.gpu_experts_mask.device,
                dtype=torch.long,
            )
            self.gpu_experts_mask[expert_indices] = True

        for op, resolved in resolved_batch:
            expert_idx = int(op.expert_idx)
            expert_key = str(expert_idx)
            source = str(resolved.get("source") or "cold")
            self.activation_applied += 1
            self._set_residency(expert_idx, ExpertResidency.GPU)
            source_counts[source] += 1
            if pre_resolved_batch is not None:
                self.apply_commit_ready_hits += 1
            if source == "activated":
                self.pipeline_promotion_source_activated += 1
            elif source == "warm":
                self.pipeline_promotion_source_warm += 1
            else:
                self.pipeline_promotion_source_cold += 1
            self.decode_prefetch_hits += 1
            self.prefetch_materialized += 1
            self.applied_migration_ops += 1
            applied += 1
            if source != "cold":
                self.pipeline_prefetch_overlap_hits += 1
            self.applied_migration_history.append(
                {
                    "phase": phase,
                    "expert_idx": expert_idx,
                    "src": op.src.value,
                    "dst": op.dst.value,
                    "reason": op.reason,
                }
            )
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                expert_idx,
                src=op.src.value,
                dst=op.dst.value,
                reason=op.reason,
                phase=phase,
                state=MigrationLifecycle.APPLIED,
            )
            completed_expert_ids.add(expert_idx)

        if applied > 0:
            self.pipeline_apply_batches += 1
            self.pipeline_apply_batch_experts += applied
        self.pipeline_apply_batch_activated += source_counts["activated"]
        self.pipeline_apply_batch_warm += source_counts["warm"]
        self.pipeline_apply_batch_cold += source_counts["cold"]
        self._update_cold_promotion_penalty(
            cold_promotions=source_counts["cold"],
            total_promotions=applied,
        )
        return applied, completed_expert_ids, source_counts

    def _demote_expert_from_gpu(self, expert_idx: int, dst: ExpertResidency) -> bool:
        expert_key = str(expert_idx)
        if expert_key in self.gpu_experts:
            expert_module = self.gpu_experts[expert_key]
            del self.gpu_experts[expert_key]
            self._store_warm_module(expert_idx, expert_module, count_store=True)
        self.gpu_experts_mask[expert_idx] = False
        self._set_residency(expert_idx, dst)
        # Notify backend to clean up DPU-resident weights if going to PIM
        if dst == ExpertResidency.PIM:
            self.offload_backend.notify_expert_evicted(expert_idx, 'gpu')
        return True

    def _coalesce_migration_ops(self, queued_ops: list) -> list:
        latest_by_expert: dict[int, object] = {}
        ordered_expert_ids: list[int] = []
        for op in queued_ops:
            expert_idx = int(op.expert_idx)
            if expert_idx in latest_by_expert:
                ordered_expert_ids.remove(expert_idx)
            latest_by_expert[expert_idx] = op
            ordered_expert_ids.append(expert_idx)
        return [latest_by_expert[expert_idx] for expert_idx in ordered_expert_ids]

    def _select_ready_promotion_batch(self, promotion_ops: list, *, max_promotions: int) -> tuple[list, int]:
        if not promotion_ops:
            return [], 0
        selected = promotion_ops[:max_promotions]
        deferred = max(0, len(promotion_ops) - len(selected))
        return selected, deferred

    def _evict_for_promotion_batch(
        self,
        *,
        protected_experts: set[int],
        fallback_dst: ExpertResidency,
        required_slots: int,
        phase: str,
    ) -> int:
        if required_slots <= 0:
            return 0

        evicted = 0
        for _ in range(required_slots):
            victim = self._pick_eviction_candidate(protected_experts, fallback_dst)
            if victim is None:
                break
            victim_idx, victim_dst = victim
            self._demote_expert_from_gpu(victim_idx, victim_dst)
            self.runtime_evictions += 1
            self.applied_migration_ops += 1
            self.pipeline_apply_batch_evictions += 1
            self.applied_migration_history.append(
                {
                    "phase": phase,
                    "expert_idx": victim_idx,
                    "src": ExpertResidency.GPU.value,
                    "dst": victim_dst.value,
                    "reason": "evict_for_ready_promotion",
                }
            )
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                victim_idx,
                src=ExpertResidency.GPU.value,
                dst=victim_dst.value,
                reason="evict_for_ready_promotion",
                phase=phase,
                state=MigrationLifecycle.APPLIED,
            )
            evicted += 1
        return evicted

    def _pick_eviction_candidate(
        self,
        protected_experts: set[int],
        fallback_dst: ExpertResidency,
    ) -> Optional[tuple[int, ExpertResidency]]:
        gpu_residents = self._runtime_gpu_residents()
        candidates = [expert_idx for expert_idx in gpu_residents if expert_idx not in protected_experts]
        if not candidates:
            return None

        hotness = None
        last_change = None
        if self.residency_plan is not None:
            state = self.residency_plan.layer_state(self.layer_idx)
            hotness = state.hotness
            last_change = state.last_residency_change_step
        current_step = 0 if self.residency_plan is None else self.residency_plan.layer_state(self.layer_idx).logical_step
        cooldown_steps = 0
        if self.dynamic_expert_scheduler is not None:
            cooldown_steps = int(self.dynamic_expert_scheduler.config.migration_cooldown_steps)
        cooled_candidates = candidates
        if last_change is not None and cooldown_steps > 0:
            cooled_candidates = [
                expert_idx
                for expert_idx in candidates
                if current_step - int(last_change[expert_idx].item()) >= cooldown_steps
            ]
            if not cooled_candidates:
                self.runtime_skipped_demotion_cooldown += len(candidates)
                return None

        if hotness is not None and hotness.numel() > 0:
            coldest = min(cooled_candidates, key=lambda expert_idx: float(hotness[expert_idx].item()))
        else:
            coldest = min(cooled_candidates)
        return coldest, fallback_dst

    def _promotion_sort_key(self, op, active_experts: set[int], phase: str) -> tuple[int, int, float, int]:
        hotness_score = self._hotness_score(int(op.expert_idx))
        is_active = 1 if int(op.expert_idx) in active_experts else 0
        is_ready = 1 if phase == "decode" and self.materialization_manager.is_ready(self.layer_idx, int(op.expert_idx)) else 0
        lifecycle_priority = self._migration_state_priority(int(op.expert_idx))
        return (-lifecycle_priority, -is_ready, -is_active, -hotness_score, int(op.expert_idx))

    def _apply_queued_migrations(
        self,
        hidden_states: torch.Tensor,
        active_experts: set[int],
        *,
        phase: str,
    ) -> None:
        if (
            phase != "decode"
            or self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
        ):
            return
        if self.pipeline_ticks > 0:
            return

        require_prefetch_ready = (
            phase == "decode"
            and self.dynamic_expert_scheduler is not None
            and self.dynamic_expert_scheduler.config.decode_require_prefetch_ready
        )
        if require_prefetch_ready:
            self._prime_pending_promotions(phase=phase)
        queued_ops = self.offload_backend.migration_manager.peek_layer(self.layer_idx)
        if not queued_ops:
            return

        queued_ops = self._coalesce_migration_ops(queued_ops)
        max_promotions = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        gpu_budget = self._runtime_gpu_budget()
        applied = 0
        applied_promotions = 0
        device = hidden_states.device
        dtype = hidden_states.dtype
        completed_expert_ids: set[int] = set()
        promotion_ops = [op for op in queued_ops if op.dst == ExpertResidency.GPU]
        demotion_ops = [
            op for op in queued_ops if op.src == ExpertResidency.GPU and op.dst != ExpertResidency.GPU
        ]
        promotion_ops.sort(key=lambda op: self._promotion_sort_key(op, active_experts, phase))

        for op in demotion_ops:
            expert_idx = int(op.expert_idx)
            if expert_idx in active_experts:
                continue
            applied_ok = self._demote_expert_from_gpu(expert_idx, op.dst)
            if applied_ok:
                applied += 1
                completed_expert_ids.add(expert_idx)
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.APPLIED,
                )
                self.applied_migration_history.append(
                    {
                        "phase": phase,
                        "expert_idx": op.expert_idx,
                        "src": op.src.value,
                        "dst": op.dst.value,
                        "reason": op.reason,
                    }
                )

        protected_experts = set(active_experts)
        protected_experts.update(int(op.expert_idx) for op in promotion_ops)

        for op in promotion_ops:
            if applied_promotions >= max_promotions:
                continue

            expert_idx = int(op.expert_idx)
            if bool(self.gpu_experts_mask[expert_idx].item()):
                completed_expert_ids.add(expert_idx)
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.APPLIED,
                )
                continue

            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            is_ready = self.materialization_manager.is_ready(self.layer_idx, expert_idx)
            ready_for_decode = lifecycle in {
                MigrationLifecycle.READY,
                MigrationLifecycle.WARMED,
                MigrationLifecycle.ACTIVATED,
            }
            if is_ready and not require_prefetch_ready:
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.READY,
                )
                ready_for_decode = True
            if require_prefetch_ready and not ready_for_decode:
                self.runtime_deferred_for_prefetch += 1
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    expert_idx,
                    src=op.src.value,
                    dst=op.dst.value,
                    reason=op.reason,
                    phase=phase,
                    state=MigrationLifecycle.DEFERRED,
                )
                if not is_ready:
                    self._request_prefetch(expert_idx)
                continue

            while gpu_budget > 0 and int(self.gpu_experts_mask.bool().sum().item()) >= gpu_budget:
                fallback_dst = (
                    self.dynamic_expert_scheduler.config.offload_tier
                    if self.dynamic_expert_scheduler is not None
                    else ExpertResidency.PIM
                )
                victim = self._pick_eviction_candidate(protected_experts, fallback_dst)
                if victim is None:
                    break
                victim_idx, victim_dst = victim
                self._demote_expert_from_gpu(victim_idx, victim_dst)
                self.runtime_evictions += 1
                applied += 1
                completed_expert_ids.add(victim_idx)
                self.offload_backend.migration_manager.mark_state(
                    self.layer_idx,
                    victim_idx,
                    src=ExpertResidency.GPU.value,
                    dst=victim_dst.value,
                    reason="evict_for_promotion",
                    phase=phase,
                    state=MigrationLifecycle.APPLIED,
                )
                self.applied_migration_history.append(
                    {
                        "phase": phase,
                        "expert_idx": victim_idx,
                        "src": ExpertResidency.GPU.value,
                        "dst": victim_dst.value,
                        "reason": "evict_for_promotion",
                    }
                )
            else:
                if is_ready:
                    self.decode_prefetch_hits += 1
                else:
                    self.decode_prefetch_misses += 1
                applied_ok = self._promote_expert_to_gpu(expert_idx, device, dtype)
                if applied_ok:
                    applied += 1
                    applied_promotions += 1
                    completed_expert_ids.add(expert_idx)
                    self.prefetch_materialized += 1
                    self.offload_backend.migration_manager.mark_state(
                        self.layer_idx,
                        expert_idx,
                        src=op.src.value,
                        dst=op.dst.value,
                        reason=op.reason,
                        phase=phase,
                        state=MigrationLifecycle.APPLIED,
                    )
                    self.applied_migration_history.append(
                        {
                            "phase": phase,
                            "expert_idx": op.expert_idx,
                            "src": op.src.value,
                            "dst": op.dst.value,
                            "reason": op.reason,
                        }
                    )
                continue

            continue

        self.applied_migration_history = self.applied_migration_history[-64:]

        if completed_expert_ids:
            self.offload_backend.migration_manager.take_layer(
                self.layer_idx,
                lambda queued_op: int(queued_op.expert_idx) in completed_expert_ids,
            )

        if applied:
            self.applied_migration_ops += applied
            self.last_applied_migration_phase = phase
            self._synchronize_gpu_mask()

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """
        前向计算。

        Args:
            hidden_states: [batch * seq_len, hidden_size]
            router_logits: [batch * seq_len, num_experts]  来自 gate 的原始 logits

        Returns:
            output: [batch * seq_len, hidden_size]  加权混合后的专家输出
        """
        # ===== Step 1: 路由 =====
        if self.router_use_softmax:
            router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
            topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_weights, dim=-1)

        context = get_context()
        phase = "prefill" if context.is_prefill else "decode"
        active_experts = {int(expert_idx) for expert_idx in torch.unique(topk_ids).tolist()}
        with self._pipeline_lock:
            self._apply_queued_migrations(hidden_states, active_experts, phase=phase)

            if self.dynamic_expert_scheduler is not None and self.dynamic_expert_scheduler.enabled:
                self.dynamic_expert_scheduler.observe(self.layer_idx, topk_ids, phase=phase)
                self._request_prefetch_candidates(phase=phase)
                planned_ops = self.dynamic_expert_scheduler.plan_layer(self.layer_idx, phase=phase)
                if planned_ops and self.offload_backend is not None:
                    for op in planned_ops:
                        if (
                            op.dst == ExpertResidency.GPU
                            and not self.materialization_manager.has_cached(self.layer_idx, int(op.expert_idx))
                        ):
                            self._request_prefetch(op.expert_idx)
                    # 当前系统只有迁移控制面，没有真实 GPU/PIM 权重迁移数据面。
                    # 这里先排队；decode 阶段会消费队列并执行最小 GPU materialize/demote。
                    self.offload_backend.queue_migration_plan(planned_ops, phase=phase)
            gpu_experts_mask_snapshot = self.gpu_experts_mask.detach().clone()
            gpu_experts_snapshot = dict(self.gpu_experts.items())

        batch_seq_len, hidden_dim = hidden_states.shape

        # ===== Step 2: 提交 CPU 专家（异步） =====
        cuda_stream = None
        if self.offload_backend is not None:
            if hidden_states.is_cuda and torch.cuda.is_available():
                cuda_stream = torch.cuda.current_stream().cuda_stream
            self.offload_backend.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

        # ===== Step 3: 并行执行 GPU 专家 =====
        final_gpu_states = torch.zeros_like(hidden_states)

        # 构建专家掩码: [batch_seq_len, num_experts]
        expert_mask = torch.nn.functional.one_hot(
            topk_ids, num_classes=self.num_experts
        ).sum(dim=1).bool()

        for expert_idx in range(self.num_experts):
            if not gpu_experts_mask_snapshot[expert_idx]:
                continue  # CPU 专家，跳过

            expert_key = str(expert_idx)
            if expert_key not in gpu_experts_snapshot:
                continue

            # 找到路由到这个专家的 token
            token_indices = torch.where(expert_mask[:, expert_idx])[0]
            if len(token_indices) == 0:
                continue

            # 提取对应 token 的 hidden states
            current_state = hidden_states[token_indices]

            # GPU 专家前向
            expert_output = gpu_experts_snapshot[expert_key](current_state)

            # 提取对应的路由权重
            expert_match = (topk_ids[token_indices] == expert_idx)
            weights = topk_weights[token_indices][expert_match]
            expert_output = expert_output * weights.unsqueeze(1)

            # 累加到结果
            final_gpu_states.index_add_(0, token_indices, expert_output)

        # ===== Step 4: 同步 CPU 专家结果 =====
        if self.offload_backend is not None:
            cpu_output = self.offload_backend.sync_forward(hidden_states, cuda_stream)
        else:
            cpu_output = torch.zeros_like(hidden_states)

        # ===== Step 5: 合并 (CPU 和 GPU 专家处理不同的 token-expert 对) =====
        return final_gpu_states + cpu_output

    def diagnostics(self) -> dict:
        with self._pipeline_lock:
            layer_residency = None
            pending_migrations = []
            if self.residency_plan is not None:
                state = self.residency_plan.layer_state(self.layer_idx)
                layer_residency = {
                    "gpu_experts": int(state.gpu_mask().sum().item()),
                    "pim_experts": int(state.pim_mask().sum().item()),
                    "cpu_experts": int(state.cpu_mask().sum().item()),
                    "epoch": state.epoch,
                }
                pending_migrations = [
                    {
                        "expert_idx": op.expert_idx,
                        "src": op.src.value,
                        "dst": op.dst.value,
                        "reason": op.reason,
                    }
                    for op in state.pending_ops
                ]
            return {
                "layer_idx": self.layer_idx,
                "offload_backend_name": self.offload_backend_name,
                "has_cpu_experts": self.has_cpu_experts,
                "applied_migration_ops": self.applied_migration_ops,
                "last_applied_migration_phase": self.last_applied_migration_phase,
                "runtime_gpu_experts": sorted(int(expert_idx) for expert_idx in self.gpu_experts.keys()),
                "gpu_experts_mask_sum": int(self.gpu_experts_mask.bool().sum().item()),
                "prefetch_requested": self.prefetch_requested,
                "prefetch_enqueued": self.prefetch_enqueued,
                "prefetch_materialized": self.prefetch_materialized,
                "prefetch_candidate_scans": self.prefetch_candidate_scans,
                "runtime_evictions": self.runtime_evictions,
                "runtime_skipped_demotion_cooldown": self.runtime_skipped_demotion_cooldown,
                "runtime_deferred_for_prefetch": self.runtime_deferred_for_prefetch,
                "decode_prefetch_hits": self.decode_prefetch_hits,
                "decode_prefetch_misses": self.decode_prefetch_misses,
                "pipeline_ticks": self.pipeline_ticks,
                "pipeline_ready_applied": self.pipeline_ready_applied,
                "pipeline_ready_deferred": self.pipeline_ready_deferred,
            "pipeline_prefetch_overlap_hits": self.pipeline_prefetch_overlap_hits,
            "pipeline_promotion_source_activated": self.pipeline_promotion_source_activated,
            "pipeline_promotion_source_warm": self.pipeline_promotion_source_warm,
            "pipeline_promotion_source_cold": self.pipeline_promotion_source_cold,
            "pipeline_apply_batches": self.pipeline_apply_batches,
            "pipeline_apply_batch_experts": self.pipeline_apply_batch_experts,
            "pipeline_apply_batch_evictions": self.pipeline_apply_batch_evictions,
            "pipeline_apply_batch_activated": self.pipeline_apply_batch_activated,
            "pipeline_apply_batch_warm": self.pipeline_apply_batch_warm,
            "pipeline_apply_batch_cold": self.pipeline_apply_batch_cold,
            "prepared_cache_limit": self.expert_prepared_cache_size,
            "prepared_cache_budget_backoff": self._prepared_cache_budget_backoff(),
            "effective_prepared_cache_limit": self._effective_prepared_cache_limit(),
            "prepared_cache_size": self._prepared_cache_size(),
            "effective_warm_cache_limit": self._effective_warm_cache_limit(),
            "prepared_cache_rebalance_pressure": self._prepared_cache_rebalance_pressure(),
            "prepared_cache_rebalance_pressure_step": self._prepared_cache_rebalance_pressure_step(),
            "prepared_cache_rebalance_pressure_ema": self.prepared_cache_rebalance_pressure_ema,
            "prepared_cache_rebalance_events_last_tick": self.prepared_cache_rebalance_events_last_tick,
            "prepared_cache_rebalance_evicted_warm": self.prepared_cache_rebalance_evicted_warm,
            "prepared_cache_rebalance_evicted_activated": self.prepared_cache_rebalance_evicted_activated,
            "prepared_cache_rebalance_demoted_to_warm": self.prepared_cache_rebalance_demoted_to_warm,
            "prepared_cache_rebalance_dropped_to_ready": self.prepared_cache_rebalance_dropped_to_ready,
            "prepared_cache_activation_stage_bonus": self.prepared_cache_activation_stage_bonus,
            "cold_promotion_penalty": self.cold_promotion_penalty,
            "adaptive_activation_limit": self._adaptive_activation_limit(),
            "adaptive_prebuild_limit": self._adaptive_prebuild_limit(),
            "adaptive_prefetch_pending_limit": self._adaptive_prefetch_pending_limit(phase="decode"),
            "adaptive_prefetch_candidate_budget": self._adaptive_prefetch_candidate_budget(phase="decode"),
            "adaptive_apply_commit_limit": self._adaptive_apply_commit_limit(background=False),
            "adaptive_apply_commit_batch_limit": self._adaptive_apply_commit_batch_limit(background=False),
            "warm_cache_hits": self.warm_cache_hits,
            "warm_cache_stores": self.warm_cache_stores,
            "warm_cache_evictions": self.warm_cache_evictions,
            "warm_cache_prebuilt": self.warm_cache_prebuilt,
            "warm_cache_device_transfers": self.warm_cache_device_transfers,
            "warm_cache_size": len(self.warm_expert_cache),
            "activated_cache_hits": self.activated_cache_hits,
            "activated_cache_stores": self.activated_cache_stores,
            "activated_cache_evictions": self.activated_cache_evictions,
            "activated_cache_size": len(self.activated_expert_cache),
            "activation_submitted": self.activation_submitted,
                "activation_ready": self.activation_ready,
                "activation_applied": self.activation_applied,
            "background_activation_applied": self.background_activation_applied,
            "apply_queue_size": len(self.apply_candidate_queue),
            "apply_queue_limit": self._apply_queue_limit(),
            "apply_queue_pending_experts": [int(expert_idx) for expert_idx in self.apply_candidate_queue.keys()],
            "apply_commit_queue_size": len(self.apply_commit_queue),
            "apply_commit_queue_limit": self._apply_commit_queue_limit(),
            "apply_commit_queue_utilization": (
                len(self.apply_commit_queue) / max(1, self._apply_commit_queue_limit())
            ),
            "apply_commit_queue_pending_experts": [int(expert_idx) for expert_idx in self.apply_commit_queue.keys()],
            "apply_commit_batch_queue_size": len(self.apply_commit_batch_queue),
            "apply_commit_batch_queue_limit": self._apply_commit_batch_queue_limit(),
            "apply_commit_batch_queue_utilization": (
                len(self.apply_commit_batch_queue) / max(1, self._apply_commit_batch_queue_limit())
            ),
            "apply_commit_batch_queue_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.apply_commit_batch_queue.values()
                for op, _resolved in batch_entries
            ],
            "resident_commit_batch_queue_size": len(self.resident_commit_batch_queue),
            "resident_commit_batch_queue_limit": self._resident_commit_batch_queue_limit(),
            "resident_commit_batch_queue_utilization": (
                len(self.resident_commit_batch_queue) / max(1, self._resident_commit_batch_queue_limit())
            ),
            "resident_commit_batch_queue_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.resident_commit_batch_queue.values()
                for op, _resolved in batch_entries
            ],
            "resident_commit_finalize_queue_size": len(self.resident_commit_finalize_queue),
            "resident_commit_finalize_queue_limit": self._resident_commit_finalize_queue_limit(),
            "resident_commit_finalize_queue_utilization": (
                len(self.resident_commit_finalize_queue) / max(1, self._resident_commit_finalize_queue_limit())
            ),
            "resident_commit_finalize_queue_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.resident_commit_finalize_queue.values()
                for op, _resolved in batch_entries
            ],
            "resident_commit_ready_cache_size": len(self.resident_commit_ready_cache),
            "resident_commit_ready_cache_limit": self._resident_commit_ready_cache_limit(),
            "resident_commit_ready_cache_utilization": (
                len(self.resident_commit_ready_cache) / max(1, self._resident_commit_ready_cache_limit())
            ),
            "resident_commit_ready_cache_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.resident_commit_ready_cache.values()
                for op, _resolved in batch_entries
            ],
            "resident_commit_apply_queue_size": len(self.resident_commit_apply_queue),
            "resident_commit_apply_queue_limit": self._resident_commit_apply_queue_limit(),
            "resident_commit_apply_queue_utilization": (
                len(self.resident_commit_apply_queue) / max(1, self._resident_commit_apply_queue_limit())
            ),
            "resident_commit_apply_queue_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.resident_commit_apply_queue.values()
                for op, _resolved in batch_entries
            ],
            "resident_commit_finalize_ready_queue_size": len(self.resident_commit_finalize_ready_queue),
            "resident_commit_finalize_ready_queue_limit": self._resident_commit_finalize_ready_queue_limit(),
            "resident_commit_finalize_ready_queue_utilization": (
                len(self.resident_commit_finalize_ready_queue)
                / max(1, self._resident_commit_finalize_ready_queue_limit())
            ),
            "resident_commit_finalize_ready_queue_pending_experts": [
                int(op.expert_idx)
                for batch_entries in self.resident_commit_finalize_ready_queue.values()
                for op, _resolved in batch_entries
            ],
            "apply_commit_ready_cache_size": len(self.apply_commit_ready_cache),
            "apply_queue_enqueued": self.apply_queue_enqueued,
            "apply_queue_committed": self.apply_queue_committed,
            "apply_queue_pruned": self.apply_queue_pruned,
            "apply_queue_evictions": self.apply_queue_evictions,
            "apply_queue_commit_batches": self.apply_queue_commit_batches,
            "apply_queue_commit_experts": self.apply_queue_commit_experts,
            "apply_commit_queue_enqueued": self.apply_commit_queue_enqueued,
            "apply_commit_queue_pruned": self.apply_commit_queue_pruned,
            "apply_commit_queue_evictions": self.apply_commit_queue_evictions,
            "apply_commit_batch_queue_enqueued": self.apply_commit_batch_queue_enqueued,
            "apply_commit_batch_queue_batches": self.apply_commit_batch_queue_batches,
            "apply_commit_batch_queue_committed_batches": self.apply_commit_batch_queue_committed_batches,
            "apply_commit_batch_queue_pruned": self.apply_commit_batch_queue_pruned,
            "apply_commit_batch_queue_evictions": self.apply_commit_batch_queue_evictions,
            "resident_commit_batch_queue_enqueued": self.resident_commit_batch_queue_enqueued,
            "resident_commit_batch_queue_batches": self.resident_commit_batch_queue_batches,
            "resident_commit_batch_queue_committed_batches": self.resident_commit_batch_queue_committed_batches,
            "resident_commit_batch_queue_pruned": self.resident_commit_batch_queue_pruned,
            "resident_commit_batch_queue_evictions": self.resident_commit_batch_queue_evictions,
            "resident_commit_finalize_queue_enqueued": self.resident_commit_finalize_queue_enqueued,
            "resident_commit_finalize_queue_batches": self.resident_commit_finalize_queue_batches,
            "resident_commit_finalize_queue_committed_batches": self.resident_commit_finalize_queue_committed_batches,
            "resident_commit_finalize_queue_pruned": self.resident_commit_finalize_queue_pruned,
            "resident_commit_finalize_queue_evictions": self.resident_commit_finalize_queue_evictions,
            "resident_commit_ready_cache_stores": self.resident_commit_ready_cache_stores,
            "resident_commit_ready_cache_hits": self.resident_commit_ready_cache_hits,
            "resident_commit_ready_cache_pruned": self.resident_commit_ready_cache_pruned,
            "resident_commit_ready_cache_evictions": self.resident_commit_ready_cache_evictions,
            "resident_commit_apply_queue_enqueued": self.resident_commit_apply_queue_enqueued,
            "resident_commit_apply_queue_batches": self.resident_commit_apply_queue_batches,
            "resident_commit_apply_queue_committed_batches": self.resident_commit_apply_queue_committed_batches,
            "resident_commit_apply_queue_pruned": self.resident_commit_apply_queue_pruned,
            "resident_commit_apply_queue_evictions": self.resident_commit_apply_queue_evictions,
            "resident_commit_finalize_ready_queue_enqueued": self.resident_commit_finalize_ready_queue_enqueued,
            "resident_commit_finalize_ready_queue_batches": self.resident_commit_finalize_ready_queue_batches,
            "resident_commit_finalize_ready_queue_committed_batches": self.resident_commit_finalize_ready_queue_committed_batches,
            "resident_commit_finalize_ready_queue_pruned": self.resident_commit_finalize_ready_queue_pruned,
            "resident_commit_finalize_ready_queue_evictions": self.resident_commit_finalize_ready_queue_evictions,
            "apply_commit_ready_hits": self.apply_commit_ready_hits,
            "apply_commit_ready_stores": self.apply_commit_ready_stores,
            "apply_commit_ready_pruned": self.apply_commit_ready_pruned,
            "background_apply_commit_resolved": self.background_apply_commit_resolved,
            "apply_queue_pressure": self._apply_queue_pressure(),
            "apply_queue_pressure_step": self._apply_queue_pressure_step(),
            "apply_queue_pressure_ema": self.apply_queue_pressure_ema,
            "apply_queue_budget_backoff": self._apply_queue_budget_backoff(),
            "apply_commit_queue_pressure": self._apply_commit_queue_pressure(),
            "apply_commit_queue_pressure_step": self._apply_commit_queue_pressure_step(),
            "apply_commit_queue_pressure_ema": self.apply_commit_queue_pressure_ema,
            "apply_commit_queue_budget_backoff": self._apply_commit_queue_budget_backoff(),
            "apply_commit_batch_queue_pressure": self._apply_commit_batch_queue_pressure(),
            "apply_commit_batch_queue_pressure_step": self._apply_commit_batch_queue_pressure_step(),
            "apply_commit_batch_queue_pressure_ema": self.apply_commit_batch_queue_pressure_ema,
            "apply_commit_batch_queue_budget_backoff": self._apply_commit_batch_queue_budget_backoff(),
            "background_apply_queue_enqueued": self.background_apply_queue_enqueued,
            "background_apply_commit_queue_enqueued": self.background_apply_commit_queue_enqueued,
            "background_apply_commit_batch_queue_enqueued": self.background_apply_commit_batch_queue_enqueued,
            "background_apply_commit_batch_queue_committed_batches": self.background_apply_commit_batch_queue_committed_batches,
            "background_apply_commit_batch_queue_prefinalized_batches": self.background_apply_commit_batch_queue_prefinalized_batches,
            "background_resident_commit_batch_queue_enqueued": self.background_resident_commit_batch_queue_enqueued,
            "background_resident_commit_batch_queue_committed_batches": self.background_resident_commit_batch_queue_committed_batches,
            "background_resident_commit_batch_queue_prefinalized_batches": self.background_resident_commit_batch_queue_prefinalized_batches,
            "background_resident_commit_finalize_queue_enqueued": self.background_resident_commit_finalize_queue_enqueued,
            "background_resident_commit_finalize_queue_committed_batches": self.background_resident_commit_finalize_queue_committed_batches,
            "background_resident_commit_finalize_queue_prefinalized_batches": self.background_resident_commit_finalize_queue_prefinalized_batches,
            "background_resident_commit_ready_cache_stores": self.background_resident_commit_ready_cache_stores,
            "background_resident_commit_apply_queue_enqueued": self.background_resident_commit_apply_queue_enqueued,
            "background_resident_commit_apply_queue_committed_batches": self.background_resident_commit_apply_queue_committed_batches,
            "background_resident_commit_finalize_ready_queue_enqueued": self.background_resident_commit_finalize_ready_queue_enqueued,
            "background_resident_commit_finalize_ready_queue_committed_batches": self.background_resident_commit_finalize_ready_queue_committed_batches,
            "background_apply_commit_batches": self.background_apply_commit_batches,
            "background_apply_commit_experts": self.background_apply_commit_experts,
            "materialization_manager": self.materialization_manager.diagnostics(),
            "layer_residency": layer_residency,
                "pending_migrations": pending_migrations,
                "backend": None if self.offload_backend is None else self.offload_backend.diagnostics(),
            }
