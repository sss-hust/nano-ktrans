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
        self.expert_warm_cache_size = max(0, int(expert_warm_cache_size))
        self.warm_expert_cache: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.activated_expert_cache: "OrderedDict[str, nn.Module]" = OrderedDict()
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
        self.materialization_manager = ExpertMaterializationManager(
            weight_path=weight_path,
            expert_key_template=self.expert_key_template,
            expert_proj_names=self.expert_proj_names,
            max_cached_experts=expert_prefetch_cache_size,
            prefetch_workers=expert_prefetch_workers,
        )

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
            if state != MigrationLifecycle.PREFETCHING:
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
        if self.offload_backend is None:
            return 0
        if not self.materialization_manager.has_pending_or_ready():
            return 0
        ready_keys = self.materialization_manager.poll_ready()
        for layer_idx, expert_idx in ready_keys:
            if int(layer_idx) != int(self.layer_idx):
                continue
            self.offload_backend.migration_manager.mark_state(
                self.layer_idx,
                int(expert_idx),
                state=MigrationLifecycle.READY,
            )
        return len(ready_keys)

    def _store_warm_module(self, expert_idx: int, module: nn.Module, *, count_store: bool) -> None:
        if self.expert_warm_cache_size <= 0:
            return
        expert_key = str(expert_idx)
        self.warm_expert_cache[expert_key] = module.to(device="cpu")
        self.warm_expert_cache.move_to_end(expert_key)
        if count_store:
            self.warm_cache_stores += 1
        while len(self.warm_expert_cache) > self.expert_warm_cache_size:
            self.warm_expert_cache.popitem(last=False)
            self.warm_cache_evictions += 1

    def _activated_cache_limit(self) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return 1
        return max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))

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

    def _activation_target_ids(self) -> set[int]:
        if self.offload_backend is None:
            return set()
        limit = self._activated_cache_limit()
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
        limit = max(1, self._activated_cache_limit() * 2)
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
            evicted_key, evicted_module = self.activated_expert_cache.popitem(last=False)
            self._store_warm_module(int(evicted_key), evicted_module, count_store=False)
            self.activated_cache_evictions += 1

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
    ) -> tuple[int, int]:
        if (
            self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
            or phase != "decode"
        ):
            return 0, 0

        max_promotions = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        queued_ops = self.offload_backend.migration_manager.peek_layer(self.layer_idx)
        if not queued_ops:
            return 0, 0

        queued_ops = self._coalesce_migration_ops(queued_ops)
        promotion_ops = []
        for op in queued_ops:
            if op.dst != ExpertResidency.GPU:
                continue
            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, int(op.expert_idx))
            if lifecycle not in {
                MigrationLifecycle.READY,
                MigrationLifecycle.WARMED,
                MigrationLifecycle.ACTIVATED,
            }:
                continue
            promotion_ops.append(op)
        if not promotion_ops:
            return 0, 0

        promotion_ops.sort(key=lambda op: self._promotion_sort_key(op, set(), phase))
        promotion_ops, deferred = self._select_ready_promotion_batch(
            promotion_ops,
            max_promotions=max_promotions,
        )
        if not promotion_ops:
            return 0, deferred
        gpu_budget = self._runtime_gpu_budget()
        protected_experts = {int(op.expert_idx) for op in promotion_ops}
        resident_ops = [op for op in promotion_ops if bool(self.gpu_experts_mask[int(op.expert_idx)].item())]
        pending_ops = [op for op in promotion_ops if not bool(self.gpu_experts_mask[int(op.expert_idx)].item())]

        applied = 0
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

        for op in pending_ops:
            expert_idx = int(op.expert_idx)
            source = self._promote_expert_to_gpu(expert_idx, device, dtype)
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
        return applied, deferred

    def advance_offload_pipeline(
        self,
        *,
        phase: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, int]:
        prefetch_submitted = self._prime_pending_promotions(phase=phase)
        ready_polled = self.refresh_offload_state()
        warm_prebuilt = self._prebuild_ready_experts(phase=phase, device=device, dtype=dtype)
        activation_ready = self._activate_warmed_experts(phase=phase, device=device, dtype=dtype)
        ready_applied = 0
        ready_deferred = 0
        if phase == "decode":
            ready_applied, ready_deferred = self._promote_ready_migrations(
                device=device,
                dtype=dtype,
                phase=phase,
            )
        self.pipeline_ticks += 1
        self.pipeline_ready_applied += ready_applied
        self.pipeline_ready_deferred += ready_deferred
        return {
            "ready_polled": int(ready_polled),
            "ready_applied": int(ready_applied),
            "ready_deferred": int(ready_deferred),
            "prefetch_submitted": int(prefetch_submitted),
            "warm_prebuilt": int(warm_prebuilt),
            "activation_ready": int(activation_ready),
            "apply_batch_count": int(self.pipeline_apply_batches),
            "apply_batch_experts": int(self.pipeline_apply_batch_experts),
            "apply_batch_evictions": int(self.pipeline_apply_batch_evictions),
        }

    def _request_prefetch_candidates(self, *, phase: str) -> None:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return
        candidates = self.dynamic_expert_scheduler.prefetch_candidates_layer(self.layer_idx, phase=phase)
        if not candidates:
            return
        self.prefetch_candidate_scans += 1
        for expert_idx in candidates:
            if not self.materialization_manager.has_cached(self.layer_idx, int(expert_idx)):
                self._request_prefetch(int(expert_idx))

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
            activated_module = self.activated_expert_cache.pop(expert_key, None)
            if activated_module is not None:
                self.activated_cache_hits += 1
                self.gpu_experts[expert_key] = activated_module.to(dtype=dtype)
                source = "activated"
            else:
                warm_module = self.warm_expert_cache.pop(expert_key, None)
                if warm_module is not None:
                    self.warm_cache_hits += 1
                    module_device = next(iter(warm_module.parameters())).device
                    if module_device != device:
                        self.gpu_experts[expert_key] = self._activate_warm_module(warm_module, device, dtype)
                    else:
                        self.gpu_experts[expert_key] = warm_module.to(dtype=dtype)
                    source = "warm"
                else:
                    self.gpu_experts[expert_key] = self._build_runtime_expert(expert_idx, device, dtype)
                    source = "cold"
        self.activation_applied += 1
        self.gpu_experts_mask[expert_idx] = True
        self._set_residency(expert_idx, ExpertResidency.GPU)
        return source

    def _demote_expert_from_gpu(self, expert_idx: int, dst: ExpertResidency) -> bool:
        expert_key = str(expert_idx)
        if expert_key in self.gpu_experts:
            expert_module = self.gpu_experts[expert_key]
            del self.gpu_experts[expert_key]
            self._store_warm_module(expert_idx, expert_module, count_store=True)
        self.gpu_experts_mask[expert_idx] = False
        self._set_residency(expert_idx, dst)
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
        if selected:
            self.pipeline_apply_batches += 1
            self.pipeline_apply_batch_experts += len(selected)
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
                continue

            lifecycle = self.offload_backend.migration_manager.state_for(self.layer_idx, expert_idx)
            is_ready = self.materialization_manager.is_ready(self.layer_idx, expert_idx)
            ready_for_decode = lifecycle in {
                MigrationLifecycle.READY,
                MigrationLifecycle.WARMED,
                MigrationLifecycle.ACTIVATED,
            }
            if is_ready and self.offload_backend is not None and not require_prefetch_ready:
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
                if self.offload_backend is not None:
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
                if self.offload_backend is not None:
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
            if not self.gpu_experts_mask[expert_idx]:
                continue  # CPU 专家，跳过

            expert_key = str(expert_idx)
            if expert_key not in self.gpu_experts:
                continue

            # 找到路由到这个专家的 token
            token_indices = torch.where(expert_mask[:, expert_idx])[0]
            if len(token_indices) == 0:
                continue

            # 提取对应 token 的 hidden states
            current_state = hidden_states[token_indices]

            # GPU 专家前向
            expert_output = self.gpu_experts[expert_key](current_state)

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
            "materialization_manager": self.materialization_manager.diagnostics(),
            "layer_residency": layer_residency,
            "pending_migrations": pending_migrations,
            "backend": None if self.offload_backend is None else self.offload_backend.diagnostics(),
        }
