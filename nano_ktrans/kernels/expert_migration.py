from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from nano_ktrans.utils.expert_runtime_state import ExpertMigrationOp


@dataclass
class MigrationPhaseRecord:
    phase: str
    plan_size: int
    completed: bool = False


@dataclass
class LayerMigrationQueue:
    layer_idx: int
    pending_ops: List[ExpertMigrationOp] = field(default_factory=list)
    history: List[MigrationPhaseRecord] = field(default_factory=list)

    def enqueue(self, ops: Sequence[ExpertMigrationOp], *, phase: str) -> None:
        if not ops:
            return
        self.pending_ops.extend(ops)
        self.history.append(MigrationPhaseRecord(phase=phase, plan_size=len(ops), completed=False))

    def drain(self) -> List[ExpertMigrationOp]:
        ops = list(self.pending_ops)
        self.pending_ops.clear()
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
                            "completed": record.completed,
                        }
                        for record in queue.history
                    ],
                }
                for layer_idx, queue in sorted(self._queues.items())
            ]
        }
