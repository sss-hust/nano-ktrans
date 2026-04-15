from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp


@dataclass
class MigrationPhaseRecord:
    phase: str
    plan_size: int
    deduped_plan_size: int = 0
    completed: bool = False


@dataclass
class LayerMigrationQueue:
    layer_idx: int
    pending_ops: List[ExpertMigrationOp] = field(default_factory=list)
    history: List[MigrationPhaseRecord] = field(default_factory=list)
    total_enqueued_ops: int = 0
    total_deduped_ops: int = 0
    total_drained_ops: int = 0

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

    def drain(self) -> List[ExpertMigrationOp]:
        ops = list(self.pending_ops)
        self.pending_ops.clear()
        self.total_drained_ops += len(ops)
        if self.history:
            self.history[-1].completed = True
        return ops


class ExpertMigrationManager:
    """
    迁移控制面占位实现。

    当前只记录 per-layer migration plan，并为后续真实 GPU/PIM 数据面预留统一入口。
    """

    def __init__(self) -> None:
        self._queues: Dict[int, LayerMigrationQueue] = {}

    def queue(self, layer_idx: int, ops: Sequence[ExpertMigrationOp], *, phase: str) -> None:
        if layer_idx not in self._queues:
            self._queues[layer_idx] = LayerMigrationQueue(layer_idx=layer_idx)
        self._queues[layer_idx].enqueue(ops, phase=phase)

    def drain_layer(self, layer_idx: int) -> List[ExpertMigrationOp]:
        queue = self._queues.get(layer_idx)
        if queue is None:
            return []
        return queue.drain()

    def diagnostics(self) -> dict:
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
                }
                for layer_idx, queue in sorted(self._queues.items())
            ]
        }
