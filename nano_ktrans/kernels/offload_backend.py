from __future__ import annotations

from abc import ABC, abstractmethod
from glob import glob
from typing import Any

import torch


def normalize_offload_backend_name(name: str | None) -> str:
    if not name:
        return "cpu"
    normalized = name.lower().replace("-", "_")
    aliases = {
        "cpu": "cpu",
        "pim": "pim_shadow",
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "submit_calls": self.submit_calls,
            "sync_calls": self.sync_calls,
        }
