from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .cpu_moe import CPUMoEBackend
from .offload_backend import count_visible_pim_ranks


class PIMMoEBackend(CPUMoEBackend):
    """
    Experimental PIM backend.

    Today this backend preserves inference correctness by reusing the CPU fallback
    math path while exposing PIM visibility and counters inside the main HybridMoE
    execution flow. The real DPU kernel remains a separate benchmark path for now.
    """

    backend_name = "pim_shadow"

    def __init__(
        self,
        *,
        layer_idx: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        gpu_experts_mask: torch.Tensor,
        weight_path: str,
        num_threads: int = 16,
        numa_pools: int = 1,
        chunked_prefill_size: int = 512,
        method: str = "AMXINT4",
        expert_key_template: Optional[str] = None,
        expert_proj_names: Optional[Dict[str, str]] = None,
        pim_rank_count: int = 1,
        pim_bytes_per_dpu: int = 1024 * 1024,
        pim_repetitions: int = 2,
    ):
        self.pim_rank_count = pim_rank_count
        self.pim_bytes_per_dpu = pim_bytes_per_dpu
        self.pim_repetitions = pim_repetitions
        self.visible_pim_ranks = count_visible_pim_ranks()
        self.offloaded_pairs = 0
        self.offloaded_tokens = 0
        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            weight_path=weight_path,
            num_threads=num_threads,
            numa_pools=numa_pools,
            chunked_prefill_size=chunked_prefill_size,
            method=method,
            expert_key_template=expert_key_template,
            expert_proj_names=expert_proj_names,
        )
        self.backend_name = "pim_shadow"

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int | None,
    ) -> None:
        cpu_mask = (~self.gpu_experts_mask.bool()).to(topk_ids.device)
        routed_to_offload = cpu_mask[topk_ids]
        self.offloaded_pairs += int(routed_to_offload.sum().item())
        self.offloaded_tokens += int(routed_to_offload.any(dim=1).sum().item())
        super().submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        diagnostics.update(
            {
                "backend_name": self.backend_name,
                "execution_mode": "shadow_cpu_fallback",
                "visible_pim_ranks": self.visible_pim_ranks,
                "configured_pim_ranks": self.pim_rank_count,
                "pim_bytes_per_dpu": self.pim_bytes_per_dpu,
                "pim_repetitions": self.pim_repetitions,
                "offloaded_pairs": self.offloaded_pairs,
                "offloaded_tokens": self.offloaded_tokens,
            }
        )
        return diagnostics
