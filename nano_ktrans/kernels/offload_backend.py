from __future__ import annotations

from abc import ABC, abstractmethod
from glob import glob
from typing import Any

import torch
from .expert_migration import ExpertMigrationManager


def normalize_offload_backend_name(name: str | None) -> str:
    if not name:
        return "cpu"
    normalized = name.lower().replace("-", "_")
    aliases = {
        "cpu": "cpu",
        "pim": "pim",
        "pim_shadow": "pim_shadow",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported offload backend: {name}")
    return aliases[normalized]


def count_visible_pim_ranks() -> int:
    return len(glob("/dev/dpu_rank*"))


class ExpertOffloadBackend(ABC):
    backend_name = "unknown"

    def __init__(self) -> None:
        self.submit_calls = 0
        self.sync_calls = 0
        self.migration_submit_calls = 0
        self.last_migration_plan_size = 0
        self.last_migration_phase = ""
        self.migration_manager = ExpertMigrationManager()

    @abstractmethod
    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream: int | None) -> torch.Tensor:
        raise NotImplementedError

    def update_gpu_expert_mask(self, gpu_experts_mask: torch.Tensor) -> None:
        raise NotImplementedError

    def queue_migration_plan(self, ops: list[Any], *, phase: str = "") -> None:
        self.migration_submit_calls += 1
        self.last_migration_plan_size = len(ops)
        self.last_migration_phase = phase
        if ops:
            layer_idx = getattr(ops[0], "layer_idx", -1)
            if layer_idx >= 0:
                self.migration_manager.queue(layer_idx, ops, phase=phase)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "submit_calls": self.submit_calls,
            "sync_calls": self.sync_calls,
            "migration_submit_calls": self.migration_submit_calls,
            "last_migration_plan_size": self.last_migration_plan_size,
            "last_migration_phase": self.last_migration_phase,
            "migration_manager": self.migration_manager.diagnostics(),
        }
