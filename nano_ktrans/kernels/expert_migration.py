from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Callable, Dict, List, Sequence

from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp, ExpertResidency


class MigrationLifecycle(str, Enum):
    QUEUED = "queued"
    PREFETCHING = "prefetching"
    READY = "ready"
    WARMED = "warmed"
    ACTIVATED = "activated"
    DEFERRED = "deferred"
    APPLIED = "applied"


PERSISTENT_MIGRATION_STATES = {
    MigrationLifecycle.PREFETCHING,
    MigrationLifecycle.READY,
    MigrationLifecycle.WARMED,
    MigrationLifecycle.ACTIVATED,
}

@dataclass
class MigrationPhaseRecord:
    phase: str
    plan_size: int
    deduped_plan_size: int = 0
    completed: bool = False


@dataclass
class MigrationLifecycleRecord:
    expert_idx: int
    src: str
    dst: str
    reason: str
    phase: str
    state: MigrationLifecycle


@dataclass
class LayerMigrationQueue:
    layer_idx: int
    pending_ops: List[ExpertMigrationOp] = field(default_factory=list)
    history: List[MigrationPhaseRecord] = field(default_factory=list)
    lifecycle: Dict[int, MigrationLifecycleRecord] = field(default_factory=dict)
    total_enqueued_ops: int = 0
    total_deduped_ops: int = 0
    total_drained_ops: int = 0
    total_ready_drains: int = 0
    total_prefetching_events: int = 0
    total_ready_events: int = 0
    total_warmed_events: int = 0
    total_activated_events: int = 0
    total_deferred_events: int = 0
    total_applied_events: int = 0
    total_requeue_preserved_states: int = 0
    total_stage_skips: int = 0
    total_warm_eviction_regressions: int = 0
    total_activation_eviction_regressions: int = 0

    def _queued_state_for_expert(
        self,
        expert_idx: int,
        queued_state: MigrationLifecycle,
    ) -> MigrationLifecycle:
        tracked = self.lifecycle.get(int(expert_idx))
        if tracked is None:
            return queued_state
        if queued_state in {MigrationLifecycle.QUEUED, MigrationLifecycle.DEFERRED}:
            if tracked.state == MigrationLifecycle.DEFERRED or tracked.state in PERSISTENT_MIGRATION_STATES:
                self.total_requeue_preserved_states += 1
                return tracked.state
        return queued_state

    def _dedupe_ops(self, ops: Sequence[ExpertMigrationOp]) -> List[ExpertMigrationOp]:
        latest_by_expert: Dict[int, ExpertMigrationOp] = {}
        ordered_expert_ids: List[int] = []
        for op in list(self.pending_ops) + list(ops):
            expert_idx = int(op.expert_idx)
            if expert_idx in latest_by_expert:
                ordered_expert_ids.remove(expert_idx)
            latest_by_expert[expert_idx] = op
            ordered_expert_ids.append(expert_idx)
        return [latest_by_expert[expert_idx] for expert_idx in ordered_expert_ids]

    def enqueue(self, ops: Sequence[ExpertMigrationOp], *, phase: str) -> None:
        if not ops:
            return
        self.total_enqueued_ops += len(ops)
        deduped_ops = self._dedupe_ops(ops)
        deduped_plan_size = len(deduped_ops)
        self.total_deduped_ops += max(0, len(self.pending_ops) + len(ops) - deduped_plan_size)
        self.pending_ops = deduped_ops
        self.history.append(
            MigrationPhaseRecord(
                phase=phase,
                plan_size=len(ops),
                deduped_plan_size=deduped_plan_size,
                completed=False,
            )
        )
        queued_state = MigrationLifecycle.DEFERRED if phase.endswith("_deferred") else MigrationLifecycle.QUEUED
        for op in deduped_ops:
            expert_state = self._queued_state_for_expert(op.expert_idx, queued_state)
            self.mark_state(
                op.expert_idx,
                src=op.src.value,
                dst=op.dst.value,
                reason=op.reason,
                phase=phase,
                state=expert_state,
            )

    def drain(self) -> List[ExpertMigrationOp]:
        return self.take(lambda _op: True)

    def take(self, predicate: Callable[[ExpertMigrationOp], bool]) -> List[ExpertMigrationOp]:
        selected: List[ExpertMigrationOp] = []
        remaining: List[ExpertMigrationOp] = []
        for op in self.pending_ops:
            if predicate(op):
                selected.append(op)
            else:
                remaining.append(op)
        self.pending_ops = remaining
        self.total_drained_ops += len(selected)
        if self.history and not self.pending_ops:
            self.history[-1].completed = True
        return selected

    def peek(self) -> List[ExpertMigrationOp]:
        ops = list(self.pending_ops)
        return ops

    def take_ready(self) -> List[ExpertMigrationOp]:
        selected = self.take(
            lambda op: (
                op.dst != ExpertResidency.GPU
                or (
                    self.lifecycle.get(int(op.expert_idx)) is not None
                    and self.lifecycle[int(op.expert_idx)].state
                    in {
                        MigrationLifecycle.READY,
                        MigrationLifecycle.WARMED,
                        MigrationLifecycle.ACTIVATED,
                    }
                )
            )
        )
        if selected:
            self.total_ready_drains += 1
        return selected

    def mark_state(
        self,
        expert_idx: int,
        *,
        src: str | None = None,
        dst: str | None = None,
        reason: str | None = None,
        phase: str | None = None,
        state: MigrationLifecycle,
    ) -> None:
        expert_idx = int(expert_idx)
        tracked = self.lifecycle.get(expert_idx)
        previous_state = tracked.state if tracked is not None else None
        if previous_state is not None:
            if previous_state in PERSISTENT_MIGRATION_STATES and state in {
                MigrationLifecycle.QUEUED,
                MigrationLifecycle.DEFERRED,
            }:
                self.total_stage_skips += 1
                return
        if tracked is None:
            tracked = MigrationLifecycleRecord(
                expert_idx=expert_idx,
                src=src or "",
                dst=dst or "",
                reason=reason or "",
                phase=phase or "",
                state=state,
            )
            self.lifecycle[expert_idx] = tracked
        else:
            if src is not None:
                tracked.src = src
            if dst is not None:
                tracked.dst = dst
            if reason is not None:
                tracked.reason = reason
            if phase is not None:
                tracked.phase = phase
            tracked.state = state

        if previous_state == state:
            return

        if previous_state == MigrationLifecycle.ACTIVATED and state == MigrationLifecycle.WARMED:
            self.total_activation_eviction_regressions += 1
        elif previous_state == MigrationLifecycle.WARMED and state == MigrationLifecycle.READY:
            self.total_warm_eviction_regressions += 1

        if state == MigrationLifecycle.PREFETCHING:
            self.total_prefetching_events += 1
        elif state == MigrationLifecycle.READY:
            self.total_ready_events += 1
        elif state == MigrationLifecycle.WARMED:
            self.total_warmed_events += 1
        elif state == MigrationLifecycle.ACTIVATED:
            self.total_activated_events += 1
        elif state == MigrationLifecycle.DEFERRED:
            self.total_deferred_events += 1
        elif state == MigrationLifecycle.APPLIED:
            self.total_applied_events += 1

    def state_for(self, expert_idx: int) -> MigrationLifecycle | None:
        tracked = self.lifecycle.get(int(expert_idx))
        return None if tracked is None else tracked.state

    def lifecycle_state_counts(self) -> Dict[str, int]:
        counts = {state.value: 0 for state in MigrationLifecycle}
        for tracked in self.lifecycle.values():
            counts[tracked.state.value] += 1
        return counts


class ExpertMigrationManager:
    """
    迁移控制面占位实现。

    当前只记录 per-layer migration plan，并为后续真实 GPU/PIM 数据面预留统一入口。
    """

    def __init__(self) -> None:
        self._queues: Dict[int, LayerMigrationQueue] = {}
        self._lock = RLock()

    def queue(self, layer_idx: int, ops: Sequence[ExpertMigrationOp], *, phase: str) -> None:
        with self._lock:
            if layer_idx not in self._queues:
                self._queues[layer_idx] = LayerMigrationQueue(layer_idx=layer_idx)
            self._queues[layer_idx].enqueue(ops, phase=phase)

    def drain_layer(self, layer_idx: int) -> List[ExpertMigrationOp]:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return []
            return queue.drain()

    def take_layer(
        self,
        layer_idx: int,
        predicate: Callable[[ExpertMigrationOp], bool],
    ) -> List[ExpertMigrationOp]:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return []
            return queue.take(predicate)

    def peek_layer(self, layer_idx: int) -> List[ExpertMigrationOp]:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return []
            return queue.peek()

    def take_ready_layer(self, layer_idx: int) -> List[ExpertMigrationOp]:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return []
            return queue.take_ready()

    def mark_state(
        self,
        layer_idx: int,
        expert_idx: int,
        *,
        state: MigrationLifecycle,
        src: str | None = None,
        dst: str | None = None,
        reason: str | None = None,
        phase: str | None = None,
    ) -> None:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return
            queue.mark_state(
                expert_idx,
                src=src,
                dst=dst,
                reason=reason,
                phase=phase,
                state=state,
            )

    def state_for(self, layer_idx: int, expert_idx: int) -> MigrationLifecycle | None:
        with self._lock:
            queue = self._queues.get(layer_idx)
            if queue is None:
                return None
            return queue.state_for(expert_idx)

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "layers": [
                    {
                        "layer_idx": layer_idx,
                        "pending_ops": len(queue.pending_ops),
                        "history": [
                            {
                                "phase": record.phase,
                                "plan_size": record.plan_size,
                                "deduped_plan_size": record.deduped_plan_size,
                                "completed": record.completed,
                            }
                            for record in queue.history
                        ],
                        "total_enqueued_ops": queue.total_enqueued_ops,
                        "total_deduped_ops": queue.total_deduped_ops,
                        "total_drained_ops": queue.total_drained_ops,
                        "total_ready_drains": queue.total_ready_drains,
                        "total_prefetching_events": queue.total_prefetching_events,
                        "total_ready_events": queue.total_ready_events,
                        "total_warmed_events": queue.total_warmed_events,
                        "total_activated_events": queue.total_activated_events,
                        "total_deferred_events": queue.total_deferred_events,
                        "total_applied_events": queue.total_applied_events,
                        "total_requeue_preserved_states": queue.total_requeue_preserved_states,
                        "total_stage_skips": queue.total_stage_skips,
                        "total_warm_eviction_regressions": queue.total_warm_eviction_regressions,
                        "total_activation_eviction_regressions": queue.total_activation_eviction_regressions,
                        "lifecycle_state_counts": queue.lifecycle_state_counts(),
                        "lifecycle": [
                            {
                                "expert_idx": tracked.expert_idx,
                                "src": tracked.src,
                                "dst": tracked.dst,
                                "reason": tracked.reason,
                                "phase": tracked.phase,
                                "state": tracked.state.value,
                            }
                            for _, tracked in sorted(queue.lifecycle.items())
                        ],
                    }
                    for layer_idx, queue in sorted(self._queues.items())
                ]
            }
