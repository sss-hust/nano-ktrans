from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from nano_ktrans.utils.expert_runtime_state import (
    ExpertMigrationOp,
    ExpertResidency,
    ExpertResidencyPlan,
    propose_topk_promotions,
    update_hotness,
)


@dataclass
class SchedulerConfig:
    enabled: bool = False
    gpu_budget_per_layer: int = 0
    hotness_decay: float = 0.95
    offload_tier: ExpertResidency = ExpertResidency.PIM
    prefill_force_gpu_budget_per_layer: int = 0
    prefill_offload_threshold_tokens: int = 8
    decode_promote_k: int = 2
    demotion_idle_steps: int = 0
    migration_cooldown_steps: int = 0
    prefill_collect_only: bool = True
    step_stride_prefill: int = 8
    step_stride_decode: int = 1


class DynamicExpertScheduler:
    def __init__(
        self,
        *,
        residency_plan: ExpertResidencyPlan,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.residency_plan = residency_plan
        self.config = config or SchedulerConfig()
        self.step = 0
        self.last_plan: List[ExpertMigrationOp] = []
        self.last_phase = "decode"

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def observe(self, layer_idx: int, topk_ids: torch.Tensor, *, phase: str = "decode") -> None:
        if not self.enabled:
            return
        state = self.residency_plan.layer_state(layer_idx)
        stride = self.config.step_stride_prefill if phase == "prefill" else self.config.step_stride_decode
        stride = max(1, int(stride))
        state.logical_step += stride
        state.hotness = update_hotness(state.hotness, topk_ids, decay=self.config.hotness_decay)
        state.mark_access(topk_ids, state.logical_step)
        self.last_phase = phase

    def plan_layer(self, layer_idx: int, *, phase: Optional[str] = None) -> List[ExpertMigrationOp]:
        if not self.enabled:
            return []
        state = self.residency_plan.layer_state(layer_idx)
        effective_phase = self.last_phase if phase is None else phase
        if effective_phase == "prefill" and self.config.prefill_collect_only:
            state.pending_ops = []
            return []
        gpu_budget = self.config.gpu_budget_per_layer
        if effective_phase == "prefill":
            gpu_budget = max(gpu_budget, self.config.prefill_force_gpu_budget_per_layer)
        ops = propose_topk_promotions(
            state,
            gpu_budget=gpu_budget,
            offload_source=self.config.offload_tier,
            current_step=state.logical_step,
            demotion_idle_steps=self.config.demotion_idle_steps,
            migration_cooldown_steps=self.config.migration_cooldown_steps,
        )
        state.pending_ops = list(ops)
        return ops

    def plan_all_layers(self, *, phase: Optional[str] = None) -> List[ExpertMigrationOp]:
        if not self.enabled:
            self.last_plan = []
            return []
        ops: List[ExpertMigrationOp] = []
        for layer_idx in range(len(self.residency_plan.layers)):
            ops.extend(self.plan_layer(layer_idx, phase=phase))
        self.last_plan = ops
        self.step += 1
        return ops

    def apply_plan(self, ops: Optional[List[ExpertMigrationOp]] = None) -> None:
        if not self.enabled:
            return
        effective_ops = self.last_plan if ops is None else ops
        for layer in self.residency_plan.layers:
            layer.apply_ops(effective_ops)

    def layer_gpu_mask(self, layer_idx: int) -> torch.Tensor:
        return self.residency_plan.layer_state(layer_idx).gpu_mask()

    def diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "step": self.step,
            "gpu_budget_per_layer": self.config.gpu_budget_per_layer,
            "prefill_force_gpu_budget_per_layer": self.config.prefill_force_gpu_budget_per_layer,
            "prefill_offload_threshold_tokens": self.config.prefill_offload_threshold_tokens,
            "decode_promote_k": self.config.decode_promote_k,
            "demotion_idle_steps": self.config.demotion_idle_steps,
            "migration_cooldown_steps": self.config.migration_cooldown_steps,
            "prefill_collect_only": self.config.prefill_collect_only,
            "step_stride_prefill": self.config.step_stride_prefill,
            "step_stride_decode": self.config.step_stride_decode,
            "last_phase": self.last_phase,
            "offload_tier": self.config.offload_tier.value,
            "last_plan_size": len(self.last_plan),
            "residency_plan": self.residency_plan.summary(),
        }
