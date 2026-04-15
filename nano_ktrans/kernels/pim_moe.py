from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .cpu_moe import CPUMoEBackend
from .offload_backend import count_visible_pim_ranks
from .pim_expert_runtime import PIMExpertRuntime
from .pim_linear_runtime import PIMLinearRuntime
from nano_ktrans.utils.context import get_context


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
        pim_execution_mode: str = "shadow",
        pim_profile: str = "",
        pim_max_batch_tokens: int = 1,
        pim_kernel_variant: str = "linear",
        pim_prefill_policy: str = "cpu",
        pim_prefill_token_threshold: int = 8,
    ):
        self.pim_rank_count = pim_rank_count
        self.pim_bytes_per_dpu = pim_bytes_per_dpu
        self.pim_repetitions = pim_repetitions
        self.pim_execution_mode = pim_execution_mode
        self.pim_profile = pim_profile
        self.pim_max_batch_tokens = pim_max_batch_tokens
        self.pim_kernel_variant = pim_kernel_variant
        self.pim_prefill_policy = pim_prefill_policy
        self.pim_prefill_token_threshold = pim_prefill_token_threshold
        self.visible_pim_ranks = count_visible_pim_ranks()
        self.offloaded_pairs = 0
        self.offloaded_tokens = 0
        self.real_dpu_linear_calls = 0
        self.real_dpu_expert_calls = 0
        self.real_dpu_fused_expert_calls = 0
        self.last_kernel_cycles = 0
        self.fallback_counts: dict[str, int] = {}
        self.runtime = None
        self.expert_runtime = None
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
        if self.pim_execution_mode == "real":
            self.backend_name = "pim"
            self.runtime = self._try_init_runtime()
            self.expert_runtime = self._try_init_expert_runtime()
        else:
            self.backend_name = "pim_shadow"

    def _record_fallback(self, reason: str) -> None:
        self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1

    def _try_init_runtime(self) -> Optional[PIMLinearRuntime]:
        if not self.has_cpu_experts:
            return None
        if self.visible_pim_ranks <= 0:
            self._record_fallback("no_visible_pim_ranks")
            return None
        try:
            return PIMLinearRuntime.get_shared(
                profile=self.pim_profile,
                rank_count=max(1, self.pim_rank_count),
            )
        except Exception:
            self._record_fallback("runtime_init_failed")
            return None

    def _try_init_expert_runtime(self) -> Optional[PIMExpertRuntime]:
        if not self.has_cpu_experts:
            return None
        if self.visible_pim_ranks <= 0:
            return None
        try:
            return PIMExpertRuntime.get_shared(
                profile=self.pim_profile,
                rank_count=max(1, self.pim_rank_count),
            )
        except Exception:
            self._record_fallback("expert_runtime_init_failed")
            return None

    def _run_expert_fused_on_dpu(
        self,
        states: torch.Tensor,
        cpu_slot: int,
    ) -> torch.Tensor:
        if self.expert_runtime is None:
            raise RuntimeError("PIM expert runtime is not available.")

        output = self.expert_runtime.expert(
            states,
            self._gate_proj[cpu_slot],
            self._up_proj[cpu_slot],
            self._down_proj[cpu_slot],
        )
        self.real_dpu_fused_expert_calls += 1
        self.real_dpu_expert_calls += 1
        self.last_kernel_cycles = self.expert_runtime.last_cycles()
        return output

    def _run_expert_linear_on_dpu(
        self,
        states: torch.Tensor,
        cpu_slot: int,
    ) -> torch.Tensor:
        if self.runtime is None:
            raise RuntimeError("PIM runtime is not available.")

        gate = self.runtime.linear(states, self._gate_proj[cpu_slot])
        up = self.runtime.linear(states, self._up_proj[cpu_slot])
        hidden = F.silu(gate) * up
        output = self.runtime.linear(hidden, self._down_proj[cpu_slot])
        self.real_dpu_linear_calls += 3
        self.real_dpu_expert_calls += 1
        self.last_kernel_cycles = self.runtime.last_cycles()
        return output

    def _submit_forward_real(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> bool:
        if self.runtime is None and self.expert_runtime is None:
            self._record_fallback("runtime_unavailable")
            return False

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device
        context = get_context()

        if context.is_prefill:
            if self.pim_prefill_policy == "cpu":
                self._record_fallback("prefill_force_cpu")
                return False
            if batch_size > self.pim_prefill_token_threshold:
                self._record_fallback("prefill_batch_too_large")
                return False

        if batch_size > self.pim_max_batch_tokens:
            self._record_fallback("batch_too_large")
            return False

        flat_cpu = flat.to("cpu", dtype=torch.float32)
        topk_ids_cpu = topk_ids.to("cpu", dtype=torch.long)
        topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)
        cpu_mask = ~self.gpu_experts_mask.bool()

        output = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device="cpu")

        for expert_idx in range(self.num_experts):
            if not cpu_mask[expert_idx]:
                continue

            cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
            if cpu_slot is None:
                continue

            match = topk_ids_cpu == expert_idx
            token_indices = torch.where(match.any(dim=1))[0]
            if len(token_indices) == 0:
                continue

            states = flat_cpu[token_indices]

            can_use_fused = (
                self.expert_runtime is not None
                and self.expert_runtime.supports_shape(
                    states.shape[0],
                    states.shape[1],
                    self.intermediate_size,
                    self.hidden_size,
                )
            )
            can_use_linear = (
                self.runtime is not None
                and self.runtime.supports_shape(states.shape[0], states.shape[1], self.intermediate_size)
                and self.runtime.supports_shape(states.shape[0], self.intermediate_size, self.hidden_size)
            )

            if self.pim_kernel_variant == "fused" and can_use_fused:
                try:
                    expert_output = self._run_expert_fused_on_dpu(states, cpu_slot)
                except Exception:
                    self._record_fallback("expert_fused_dpu_run_failed")
                    if can_use_linear:
                        try:
                            expert_output = self._run_expert_linear_on_dpu(states, cpu_slot)
                        except Exception:
                            self._record_fallback("expert_dpu_run_failed")
                            expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)
                    else:
                        expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)
            elif self.runtime is None or not self.runtime.supports_shape(states.shape[0], states.shape[1], self.intermediate_size):
                self._record_fallback("gate_up_shape_unsupported")
                expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)
            elif not self.runtime.supports_shape(states.shape[0], self.intermediate_size, self.hidden_size):
                self._record_fallback("down_shape_unsupported")
                expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)
            else:
                try:
                    expert_output = self._run_expert_linear_on_dpu(states, cpu_slot)
                except Exception:
                    self._record_fallback("expert_dpu_run_failed")
                    expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)

            row_idx, col_idx = torch.where(match[token_indices])
            weights = topk_weights_cpu[token_indices[row_idx], col_idx].to(dtype=expert_output.dtype).unsqueeze(1)
            output.index_add_(0, token_indices[row_idx], expert_output[row_idx] * weights)

        self._fallback_output = output.to(device=device, dtype=hidden_states.dtype)
        return True

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

        if self.pim_execution_mode == "real" and self.has_cpu_experts:
            if self._submit_forward_real(hidden_states, topk_ids, topk_weights):
                return

        super().submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

    def update_gpu_expert_mask(self, gpu_experts_mask: torch.Tensor) -> None:
        super().update_gpu_expert_mask(gpu_experts_mask)

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        diagnostics.update(
            {
                "backend_name": self.backend_name,
                "execution_mode": "dpu_linear_host_activation" if self.pim_execution_mode == "real" else "shadow_cpu_fallback",
                "runtime_available": self.runtime is not None,
                "expert_runtime_available": self.expert_runtime is not None,
                "visible_pim_ranks": self.visible_pim_ranks,
                "configured_pim_ranks": self.pim_rank_count,
                "runtime_dpu_count": 0 if self.runtime is None else self.runtime.num_dpus(),
                "expert_runtime_dpu_count": 0 if self.expert_runtime is None else self.expert_runtime.num_dpus(),
                "expert_runtime_last_active_dpus": 0
                if self.expert_runtime is None
                else self.expert_runtime.last_active_dpus(),
                "pim_bytes_per_dpu": self.pim_bytes_per_dpu,
                "pim_repetitions": self.pim_repetitions,
                "pim_profile": self.pim_profile,
                "pim_max_batch_tokens": self.pim_max_batch_tokens,
                "pim_kernel_variant": self.pim_kernel_variant,
                "pim_prefill_policy": self.pim_prefill_policy,
                "pim_prefill_token_threshold": self.pim_prefill_token_threshold,
                "offloaded_pairs": self.offloaded_pairs,
                "offloaded_tokens": self.offloaded_tokens,
                "real_dpu_linear_calls": self.real_dpu_linear_calls,
                "real_dpu_expert_calls": self.real_dpu_expert_calls,
                "real_dpu_fused_expert_calls": self.real_dpu_fused_expert_calls,
                "last_kernel_cycles": self.last_kernel_cycles,
                "fallback_counts": dict(self.fallback_counts),
            }
        )
        return diagnostics
