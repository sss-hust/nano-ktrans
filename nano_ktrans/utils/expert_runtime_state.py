from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Sequence

import torch


class ExpertResidency(str, Enum):
    GPU = "gpu"
    PIM = "pim"
    CPU = "cpu"


def _residency_to_code(residency: ExpertResidency) -> int:
    if residency == ExpertResidency.GPU:
        return 1
    if residency == ExpertResidency.PIM:
        return 2
    return 3


def _code_to_residency(code: int) -> ExpertResidency:
    if code == 1:
        return ExpertResidency.GPU
    if code == 2:
        return ExpertResidency.PIM
    return ExpertResidency.CPU


@dataclass
class ExpertMigrationOp:
    layer_idx: int
    expert_idx: int
    src: ExpertResidency
    dst: ExpertResidency
    reason: str = ""


@dataclass
class LayerPlacementDecision:
    layer_idx: int
    gpu_experts: List[int]
    pim_experts: List[int]
    cpu_experts: List[int]
    migration_ops: List[ExpertMigrationOp]


@dataclass
class LayerExpertState:
    layer_idx: int
    residency: torch.Tensor
    hotness: torch.Tensor
    last_access_step: torch.Tensor
    last_residency_change_step: torch.Tensor
    logical_step: int = 0
    epoch: int = 0
    pending_ops: List[ExpertMigrationOp] = field(default_factory=list)

    @classmethod
    def from_gpu_mask(
        cls,
        layer_idx: int,
        gpu_mask: torch.Tensor,
        offload_tier: ExpertResidency,
    ) -> "LayerExpertState":
        residency = torch.full(
            (gpu_mask.numel(),),
            fill_value=_residency_to_code(offload_tier),
            dtype=torch.uint8,
        )
        residency[gpu_mask.bool().cpu()] = _residency_to_code(ExpertResidency.GPU)
        hotness = torch.zeros(gpu_mask.numel(), dtype=torch.float32)
        last_access_step = torch.full((gpu_mask.numel(),), fill_value=-1, dtype=torch.int32)
        last_residency_change_step = torch.full((gpu_mask.numel(),), fill_value=-1, dtype=torch.int32)
        return cls(
            layer_idx=layer_idx,
            residency=residency,
            hotness=hotness,
            last_access_step=last_access_step,
            last_residency_change_step=last_residency_change_step,
        )

    def gpu_mask(self) -> torch.Tensor:
        return self.residency == _residency_to_code(ExpertResidency.GPU)

    def pim_mask(self) -> torch.Tensor:
        return self.residency == _residency_to_code(ExpertResidency.PIM)

    def cpu_mask(self) -> torch.Tensor:
        return self.residency == _residency_to_code(ExpertResidency.CPU)

    def count(self, residency: ExpertResidency) -> int:
        return int((self.residency == _residency_to_code(residency)).sum().item())

    def mark_access(self, routed_expert_ids: torch.Tensor, step: int) -> None:
        flat_ids = routed_expert_ids.reshape(-1).to(dtype=torch.long, device="cpu")
        if flat_ids.numel() == 0:
            return
        touched = torch.unique(flat_ids)
        self.last_access_step[touched] = int(step)

    def record_residency_change(
        self,
        expert_idx: int,
        dst: ExpertResidency,
        *,
        step: int,
    ) -> None:
        self.residency[expert_idx] = _residency_to_code(dst)
        self.last_residency_change_step[expert_idx] = int(step)
        self.epoch += 1
        self.pending_ops = []

    def apply_ops(self, ops: Sequence[ExpertMigrationOp], *, step: int | None = None) -> None:
        changed = False
        for op in ops:
            if op.layer_idx != self.layer_idx:
                continue
            self.residency[op.expert_idx] = _residency_to_code(op.dst)
            if step is not None:
                self.last_residency_change_step[op.expert_idx] = int(step)
            changed = True
        if changed:
            self.epoch += 1
            self.pending_ops = []


@dataclass
class ExpertResidencyPlan:
    layers: List[LayerExpertState]
    default_offload_tier: ExpertResidency = ExpertResidency.PIM

    @classmethod
    def from_gpu_masks(
        cls,
        layer_gpu_masks: Sequence[torch.Tensor],
        *,
        default_offload_tier: ExpertResidency = ExpertResidency.PIM,
    ) -> "ExpertResidencyPlan":
        return cls(
            layers=[
                LayerExpertState.from_gpu_mask(layer_idx, gpu_mask.cpu(), default_offload_tier)
                for layer_idx, gpu_mask in enumerate(layer_gpu_masks)
            ],
            default_offload_tier=default_offload_tier,
        )

    def gpu_masks(self) -> List[torch.Tensor]:
        return [layer.gpu_mask() for layer in self.layers]

    def layer_state(self, layer_idx: int) -> LayerExpertState:
        return self.layers[layer_idx]

    def summary(self) -> dict:
        return {
            "default_offload_tier": self.default_offload_tier.value,
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "gpu_experts": layer.count(ExpertResidency.GPU),
                    "pim_experts": layer.count(ExpertResidency.PIM),
                    "cpu_experts": layer.count(ExpertResidency.CPU),
                    "logical_step": layer.logical_step,
                    "epoch": layer.epoch,
                }
                for layer in self.layers
            ],
        }

    def placement_decisions(self) -> List[LayerPlacementDecision]:
        decisions: List[LayerPlacementDecision] = []
        for layer in self.layers:
            gpu_experts = torch.where(layer.gpu_mask())[0].tolist()
            pim_experts = torch.where(layer.pim_mask())[0].tolist()
            cpu_experts = torch.where(layer.cpu_mask())[0].tolist()
            decisions.append(
                LayerPlacementDecision(
                    layer_idx=layer.layer_idx,
                    gpu_experts=gpu_experts,
                    pim_experts=pim_experts,
                    cpu_experts=cpu_experts,
                    migration_ops=list(layer.pending_ops),
                )
            )
        return decisions


def update_hotness(
    hotness: torch.Tensor,
    routed_expert_ids: torch.Tensor,
    *,
    decay: float = 0.95,
    router_scores: torch.Tensor | None = None,
    mrs_alpha: float | None = None,
    top_p: int | None = None,
) -> torch.Tensor:
    """
    Update per-expert hotness EMA.

    Two modes:

    1. **Bincount mode** (``mrs_alpha is None`` or ``router_scores is None``):
       Classical behaviour — ``hotness = hotness * decay + bincount(routed_ids)``.
       Each activation counts as "+1" regardless of router confidence.

    2. **MRS mode** (HybriMoE, arXiv:2504.05897, §IV.C):
       ``S = alpha * TopP(router_scores) + (1 - alpha) * S``

       Router probability mass is folded directly into the hotness EMA, so a
       confidently routed expert contributes more than a top-k tie-breaker.
       Only the ``top_p`` experts per token are accumulated — low-probability
       experts have indistinguishable reuse signal and are clipped.

    The MRS mode is the recommended default whenever Hybrid MoE has a
    score-aware cache layer (warm / activated), because it lets cache victim
    selection fall back to a signal that is continuous in ``[0, 1]`` rather
    than a sparse integer count.

    Args:
        hotness: ``[num_experts]`` running estimate (modified out-of-place).
        routed_expert_ids: ``[N, top_k]`` (or flat) expert ids selected by gate.
        decay: Multiplicative decay for bincount mode. Ignored in MRS mode.
        router_scores: ``[N, top_k]`` probabilities paired with ``routed_expert_ids``.
            If provided together with ``mrs_alpha``, enables MRS mode.
        mrs_alpha: Smoothing coefficient in ``(0, 1]``. Typical value ``0.3``.
        top_p: Per-token truncation cap. Defaults to ``routed_expert_ids.shape[-1]``
            (i.e. no extra truncation beyond top-k). Setting ``top_p`` to
            ``2 * top_k`` matches HybriMoE's default.

    Returns:
        Updated hotness tensor (same shape/dtype as input).
    """
    num_experts = updated_numel = hotness.numel()
    if mrs_alpha is not None and router_scores is not None:
        alpha = float(mrs_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"mrs_alpha must be in (0, 1], got {alpha}")

        flat_ids = routed_expert_ids.reshape(-1, routed_expert_ids.shape[-1]).to(
            dtype=torch.long, device="cpu"
        )
        flat_scores = router_scores.reshape(-1, router_scores.shape[-1]).to(
            dtype=hotness.dtype, device="cpu"
        )
        if flat_ids.shape != flat_scores.shape:
            raise ValueError(
                f"routed_expert_ids and router_scores shape mismatch: "
                f"{tuple(flat_ids.shape)} vs {tuple(flat_scores.shape)}"
            )
        if flat_ids.numel() == 0 or num_experts == 0:
            return hotness * (1.0 - alpha)

        effective_p = int(top_p) if top_p is not None else flat_ids.shape[-1]
        effective_p = max(1, min(effective_p, flat_ids.shape[-1]))
        if effective_p < flat_ids.shape[-1]:
            top_values, top_positions = torch.topk(flat_scores, k=effective_p, dim=-1)
            top_ids = torch.gather(flat_ids, 1, top_positions)
        else:
            top_values = flat_scores
            top_ids = flat_ids

        # Aggregate per-expert score mass for this observation window.
        score_mass = torch.zeros(num_experts, dtype=hotness.dtype)
        score_mass.scatter_add_(0, top_ids.reshape(-1), top_values.reshape(-1))

        # Normalize by number of tokens so a long sequence doesn't overwhelm
        # EMA history — matches HybriMoE's "per-iteration TopP(s)" semantics.
        token_count = max(1, int(top_ids.shape[0]))
        score_mass = score_mass / float(token_count)
        return hotness * (1.0 - alpha) + score_mass * alpha

    # ── Classical bincount mode ────────────────────────────────────────
    updated = hotness * decay
    flat_ids = routed_expert_ids.reshape(-1).to(dtype=torch.long, device="cpu")
    if flat_ids.numel() > 0:
        bincount = torch.bincount(flat_ids, minlength=updated_numel).to(dtype=updated.dtype)
        updated = updated + bincount
    return updated


def propose_topk_promotions(
    layer_state: LayerExpertState,
    *,
    gpu_budget: int,
    offload_source: ExpertResidency = ExpertResidency.PIM,
    current_step: int = 0,
    demotion_idle_steps: int = 0,
    migration_cooldown_steps: int = 0,
) -> List[ExpertMigrationOp]:
    hotness = layer_state.hotness
    if hotness.numel() == 0 or gpu_budget <= 0:
        return []

    k = min(gpu_budget, hotness.numel())
    _, top_indices = torch.topk(hotness, k=k)
    desired_gpu = set(int(idx) for idx in top_indices.tolist())
    current_gpu = set(torch.where(layer_state.gpu_mask())[0].tolist())
    current_offload = set(
        torch.where(layer_state.residency == _residency_to_code(offload_source))[0].tolist()
    )

    ops: List[ExpertMigrationOp] = []
    for expert_idx in sorted(desired_gpu - current_gpu):
        if expert_idx not in current_offload:
            continue
        if current_step - int(layer_state.last_residency_change_step[expert_idx].item()) < migration_cooldown_steps:
            continue
        ops.append(
            ExpertMigrationOp(
                layer_idx=layer_state.layer_idx,
                expert_idx=expert_idx,
                src=offload_source,
                dst=ExpertResidency.GPU,
                reason="promote_hot_expert",
            )
        )
    for expert_idx in sorted(current_gpu - desired_gpu):
        if current_step - int(layer_state.last_access_step[expert_idx].item()) <= demotion_idle_steps:
            continue
        if current_step - int(layer_state.last_residency_change_step[expert_idx].item()) < migration_cooldown_steps:
            continue
        ops.append(
            ExpertMigrationOp(
                layer_idx=layer_state.layer_idx,
                expert_idx=expert_idx,
                src=ExpertResidency.GPU,
                dst=offload_source,
                reason="demote_cold_expert",
            )
        )
    return ops


def select_topk_offload_candidates(
    layer_state: LayerExpertState,
    *,
    candidate_budget: int,
    offload_source: ExpertResidency = ExpertResidency.PIM,
) -> List[int]:
    if candidate_budget <= 0 or layer_state.hotness.numel() == 0:
        return []

    offload_mask = layer_state.residency == _residency_to_code(offload_source)
    candidate_indices = torch.where(offload_mask)[0]
    if candidate_indices.numel() == 0:
        return []

    candidate_hotness = layer_state.hotness.index_select(0, candidate_indices)
    k = min(int(candidate_budget), int(candidate_indices.numel()))
    _, top_pos = torch.topk(candidate_hotness, k=k)
    return [int(candidate_indices[idx].item()) for idx in top_pos.tolist()]


def residency_codes_to_strings(residency: Iterable[int]) -> List[str]:
    return [_code_to_residency(int(code)).value for code in residency]
