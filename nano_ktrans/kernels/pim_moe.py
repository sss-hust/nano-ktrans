from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .cpu_moe import CPUMoEBackend
from .offload_backend import count_visible_pim_ranks
from .pim_expert_runtime import PIMExpertRuntime
from .pim_linear_runtime import PIMLinearRuntime
from .pim_quantized_runtime import PIMQuantizedRuntime
from nano_ktrans.utils.context import get_context

# ADR-002 M-3: cost-model-driven backend routing.  Imported lazily at
# use-time via ``_get_default_cost_model`` so shadow-mode backends that
# never touch real DPUs don't pay the table load.
_DEFAULT_COST_MODEL: Optional[object] = None


def _get_default_cost_model():
    global _DEFAULT_COST_MODEL
    if _DEFAULT_COST_MODEL is not None:
        return _DEFAULT_COST_MODEL
    try:
        from nano_ktrans.scheduler.cost_model import load_default_cost_model
    except Exception:
        return None
    _DEFAULT_COST_MODEL = load_default_cost_model()
    return _DEFAULT_COST_MODEL


class PIMMoEBackend(CPUMoEBackend):
    """
    Experimental PIM backend with weight residency optimization.

    Key optimization: expert weights are pre-loaded to DPU MRAM and persist
    across inference calls. Only input activations are transferred per call.
    Weight re-transfer only happens when switching to a different expert.
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
        cost_model: Optional[object] = None,
        enable_cost_model_routing: bool = True,
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
        # ADR-002 M-3: cost model.  If the caller didn't pass one and
        # cost-model routing is enabled, load the M-2 baseline.  A
        # failure to load (missing file, parse error) is soft: we fall
        # back to the legacy threshold behaviour.
        self.enable_cost_model_routing = bool(enable_cost_model_routing)
        if cost_model is not None:
            self.cost_model = cost_model
        elif self.enable_cost_model_routing:
            self.cost_model = _get_default_cost_model()
        else:
            self.cost_model = None
        # Per-decision counters (for diagnostics).
        self.cost_model_decisions_pim: int = 0
        self.cost_model_decisions_cpu: int = 0
        self.visible_pim_ranks = count_visible_pim_ranks()
        self.offloaded_pairs = 0
        self.offloaded_tokens = 0
        self.real_dpu_linear_calls = 0
        self.real_dpu_expert_calls = 0
        self.real_dpu_fused_expert_calls = 0
        self.real_dpu_quantized_calls = 0
        self.last_kernel_cycles = 0
        self.fallback_counts: dict[str, int] = {}
        self.runtime = None
        self.expert_runtime = None
        self.quantized_runtime: Optional[PIMQuantizedRuntime] = None
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
            if self.is_gptq:
                self.quantized_runtime = self._try_init_quantized_runtime()
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

    def _try_init_quantized_runtime(self) -> Optional[PIMQuantizedRuntime]:
        if not self.has_cpu_experts:
            return None
        if self.visible_pim_ranks <= 0:
            return None
        if not self.is_gptq:
            return None
        try:
            return PIMQuantizedRuntime.get_shared(
                profile=self.pim_profile,
                rank_count=max(1, self.pim_rank_count),
            )
        except Exception:
            self._record_fallback("quantized_runtime_init_failed")
            return None

    # ── Expert ID for weight residency tracking ────────────────────────

    def _expert_id(self, cpu_slot: int) -> int:
        """Stable expert identity for DPU residency tracking."""
        return hash((self.layer_idx, cpu_slot)) & 0xFFFFFFFFFFFFFFFF

    # ── Fused expert path (preload + infer) ────────────────────────────

    def _run_expert_fused_on_dpu(
        self,
        states: torch.Tensor,
        cpu_slot: int,
    ) -> torch.Tensor:
        if self.expert_runtime is None:
            raise RuntimeError("PIM expert runtime is not available.")

        eid = self._expert_id(cpu_slot)

        # Preload only if this expert isn't already resident
        self.expert_runtime.preload(
            eid,
            self._gate_proj[cpu_slot],
            self._up_proj[cpu_slot],
            self._down_proj[cpu_slot],
        )

        # Infer: only input activations are transferred to DPU
        output = self.expert_runtime.infer(states)

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

    # ── Quantized expert path (W4A32 preload + infer) ──────────────────

    def _run_expert_quantized_on_dpu(
        self,
        states: torch.Tensor,
        cpu_slot: int,
    ) -> torch.Tensor:
        """Run expert MLP on DPU using quantized (GPTQ) weights with preload caching.

        ADR-002 M-4.1: fuse gate+up into one DPU launch via
        ``preload_and_infer_concat``.  Three DPU launches per expert
        become two, and host->DPU weight transfer per expert drops by
        one third.
        """
        if self.quantized_runtime is None:
            raise RuntimeError("PIM quantized runtime is not available.")

        gptq = self._gptq_experts.get(cpu_slot)
        if gptq is None:
            raise RuntimeError(f"No GPTQ weights for cpu_slot={cpu_slot}")

        kernel_mode = 4  # int8 fixed-point (best for batch=1 decode)

        # Each fused call gets its own expert_id namespace so the
        # single-slot residency tracker in PIMQuantizedRuntime can
        # distinguish it from the down projection's bundle.
        base_eid = self._expert_id(cpu_slot)
        gate_up_eid = base_eid ^ 0x1212121212121212

        # Fused gate+up: one DPU launch instead of two.
        gate, up = self.quantized_runtime.preload_and_infer_concat(
            gate_up_eid,
            gptq["gate"],
            gptq["up"],
            states,
            kernel_mode=kernel_mode,
        )

        # SiLU activation on host.
        hidden = F.silu(gate) * up

        # Down projection — still its own DPU call.
        down_eid = base_eid ^ 0x3333333333333333
        self.quantized_runtime.preload(down_eid, gptq["down"], kernel_mode)
        output = self.quantized_runtime.infer(hidden)

        # Counters — 2 quantized calls per expert now (fused gate/up + down).
        self.real_dpu_quantized_calls += 2
        self.real_dpu_expert_calls += 1
        self.last_kernel_cycles = self.quantized_runtime.last_cycles()
        return output

    # ── Speculative preload at end of prefill ──────────────────────────

    def _speculative_preload(self, topk_ids: torch.Tensor) -> None:
        """
        Called at end of prefill to warm the DPU for decode.
        Preloads the most frequently routed CPU expert.
        """
        if self.expert_runtime is None:
            return

        cpu_mask = ~self.gpu_experts_mask.bool()
        flat_ids = topk_ids.view(-1).cpu().long()
        counts = torch.bincount(flat_ids, minlength=self.num_experts)
        counts[~cpu_mask] = 0  # only consider CPU experts

        if counts.max() == 0:
            return

        best_expert = int(counts.argmax())
        cpu_slot = self.cpu_expert_lookup.get(best_expert)
        if cpu_slot is None:
            return

        eid = self._expert_id(cpu_slot)
        try:
            self.expert_runtime.preload(
                eid,
                self._gate_proj[cpu_slot],
                self._up_proj[cpu_slot],
                self._down_proj[cpu_slot],
            )
        except Exception:
            self._record_fallback("speculative_preload_failed")

    # ── Real forward with expert ordering optimization ─────────────────

    def _submit_forward_real(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> bool:
        if self.runtime is None and self.expert_runtime is None and self.quantized_runtime is None:
            self._record_fallback("runtime_unavailable")
            return False

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device
        context = get_context()

        # ADR-002 M-3: cost-model-driven layer-level gate.
        #
        # The previous logic was a trio of hardcoded knobs
        # (``pim_prefill_policy``, ``pim_prefill_token_threshold``,
        # ``pim_max_batch_tokens``) that approximated "only trust PIM at
        # batch=1".  M-2 produced 120 real cells showing exactly which
        # (shape, batch, rank) combinations are faster on PIM vs CPU
        # grouped; we now query that table directly.
        #
        # Gate decision rule:
        #   * If the cost model is unavailable OR cost-model routing is
        #     disabled, keep the legacy threshold path (back-compat).
        #   * Otherwise, ask the cost model for each of the three expert
        #     projection shapes (gate/up/down).  If the model picks CPU
        #     for the majority, return False (drop to CPU-AMX fallback).
        #     If it picks PIM for at least one projection, go through
        #     the existing per-expert loop — the per-expert code already
        #     handles fine-grained dispatch (quantized / fused / linear /
        #     cpu fallback) so we don't need to re-route here.
        use_cost_model = (
            self.cost_model is not None
            and self.enable_cost_model_routing
            and self.is_gptq  # M-2 baseline only covers GPTQ quantized experts
        )
        if use_cost_model:
            decisions = []
            for shape_name in ("gate", "up", "down"):
                try:
                    decision = self.cost_model.decide(
                        shape_name=shape_name,
                        batch=batch_size,
                        rank_count=self.pim_rank_count,
                        is_prefill=context.is_prefill,
                        pim_available=True,
                    )
                except Exception:
                    decision = None
                decisions.append(decision)
            picks = [
                d.backend for d in decisions if d is not None and d.backend in ("pim", "cpu")
            ]
            pim_votes = sum(1 for p in picks if p == "pim")
            cpu_votes = sum(1 for p in picks if p == "cpu")
            if picks and cpu_votes > pim_votes:
                # Majority of projections predicted cheaper on CPU — drop
                # to CPU-AMX/grouped entirely.
                self._record_fallback("cost_model_prefers_cpu")
                self.cost_model_decisions_cpu += 1
                return False
            # Else PIM wins (or tied) — fall through and execute per-expert
            # loop.  Record the decision.
            self.cost_model_decisions_pim += 1
        else:
            # Legacy hard-threshold path (pre-M-3).  Preserved so
            # production deployments that predate the cost model still
            # run deterministically.
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

        # ── Collect activated CPU experts ──────────────────────────────
        activated_cpu_experts = []
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

            activated_cpu_experts.append((expert_idx, cpu_slot, token_indices, match))

        # ── Sort: currently resident expert goes LAST (stays in MRAM) ──
        resident_eid = 0
        if self.quantized_runtime is not None:
            resident_eid = self.quantized_runtime.resident_expert_id
        elif self.expert_runtime is not None:
            resident_eid = self.expert_runtime.resident_expert_id
        activated_cpu_experts.sort(
            key=lambda x: (1 if self._expert_id(x[1]) == resident_eid else 0,)
        )

        # ── Process each expert ────────────────────────────────────────
        for expert_idx, cpu_slot, token_indices, match in activated_cpu_experts:
            states = flat_cpu[token_indices]

            # Priority 1: Quantized DPU path (GPTQ weights)
            can_use_quantized = (
                self.quantized_runtime is not None
                and self.is_gptq
                and cpu_slot in self._gptq_experts
            )

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

            if can_use_quantized:
                try:
                    expert_output = self._run_expert_quantized_on_dpu(states, cpu_slot)
                except Exception:
                    self._record_fallback("expert_quantized_dpu_run_failed")
                    expert_output = self._compute_expert_output_cpu(states, cpu_slot).to(dtype=torch.float32)
            elif self.pim_kernel_variant == "fused" and can_use_fused:
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

        context = get_context()

        if self.pim_execution_mode == "real" and self.has_cpu_experts:
            if self._submit_forward_real(hidden_states, topk_ids, topk_weights):
                # If prefill fell through to CPU, speculatively preload for decode
                return

        # Prefill path: after CPU fallback, speculatively preload hottest expert
        if context.is_prefill and self.pim_execution_mode == "real" and self.expert_runtime is not None:
            self._speculative_preload(topk_ids)

        super().submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

    def update_gpu_expert_mask(self, gpu_experts_mask: torch.Tensor) -> None:
        super().update_gpu_expert_mask(gpu_experts_mask)

    def notify_expert_evicted(self, expert_idx: int, residency_before: str) -> None:
        """
        Clean up DPU-resident weights when an expert is evicted from GPU to PIM/CPU.
        """
        cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
        if cpu_slot is None:
            return

        eid = self._expert_id(cpu_slot)

        # Clean up fp32 expert runtime
        if self.expert_runtime is not None:
            try:
                if self.expert_runtime.resident_expert_id == eid:
                    self.expert_runtime.evict()
                self.expert_runtime.evict_cached_weights(eid)
            except Exception:
                pass

        # Clean up quantized runtime (all projection bundle IDs:
        # legacy gate/up/down + ADR-002 M-4.1 fused gate+up).
        if self.quantized_runtime is not None:
            try:
                for xor_mask in (
                    0x1111111111111111,
                    0x2222222222222222,
                    0x3333333333333333,
                    0x1212121212121212,  # M-4.1 fused gate+up bundle
                ):
                    proj_eid = eid ^ xor_mask
                    self.quantized_runtime.evict_cached_weights(proj_eid)
            except Exception:
                pass

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
                # ADR-002 M-3: cost-model routing diagnostics.
                "cost_model_enabled": (
                    self.cost_model is not None and self.enable_cost_model_routing
                ),
                "cost_model_routing_flag": self.enable_cost_model_routing,
                "cost_model_decisions_pim": self.cost_model_decisions_pim,
                "cost_model_decisions_cpu": self.cost_model_decisions_cpu,
                "cost_model_state": (
                    self.cost_model.diagnostics()
                    if (self.cost_model is not None and hasattr(self.cost_model, "diagnostics"))
                    else None
                ),
                "offloaded_pairs": self.offloaded_pairs,
                "offloaded_tokens": self.offloaded_tokens,
                "real_dpu_linear_calls": self.real_dpu_linear_calls,
                "real_dpu_expert_calls": self.real_dpu_expert_calls,
                "real_dpu_fused_expert_calls": self.real_dpu_fused_expert_calls,
                "real_dpu_quantized_calls": self.real_dpu_quantized_calls,
                "last_kernel_cycles": self.last_kernel_cycles,
                "fallback_counts": dict(self.fallback_counts),
                "is_gptq": self.is_gptq,
                # Weight residency diagnostics (fp32 expert runtime)
                "expert_runtime_preload_hits": 0 if self.expert_runtime is None else self.expert_runtime.preload_hits,
                "expert_runtime_preload_misses": 0 if self.expert_runtime is None else self.expert_runtime.preload_misses,
                "expert_runtime_resident_expert_id": 0 if self.expert_runtime is None else self.expert_runtime.resident_expert_id,
                "expert_runtime_weight_cache_size": 0 if self.expert_runtime is None else len(self.expert_runtime._weight_cache),
                # Weight residency diagnostics (quantized runtime)
                "quantized_runtime_available": self.quantized_runtime is not None,
                "quantized_runtime_preload_hits": 0 if self.quantized_runtime is None else self.quantized_runtime.preload_hits,
                "quantized_runtime_preload_misses": 0 if self.quantized_runtime is None else self.quantized_runtime.preload_misses,
                "quantized_runtime_resident_expert_id": 0 if self.quantized_runtime is None else self.quantized_runtime.resident_expert_id,
                "quantized_runtime_weight_cache_size": 0 if self.quantized_runtime is None else len(self.quantized_runtime._weight_cache),
            }
        )
        return diagnostics
