from __future__ import annotations

import threading
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
        pim_layer_group_size: int = 48,
        enable_speculative_preload_gptq: bool = False,
        enable_async_pim_submit: bool = False,
        enable_c_fused_kernel: bool = False,
        enable_c_async_submit: bool = False,
        enable_m25_pinned_d2h: bool = False,
        enable_m26_threaded_submit: bool = False,
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
        # ADR-002 M-7: layer-group scoping of PIMQuantizedRuntime slot
        # tables.  With group_size=3 on Qwen3 (48 layers), 16 disjoint
        # (gate_up, down) runtime pairs exist — each holds its own
        # 8-slot LRU and so sees a 3×8 = 24 unique-expert working set,
        # giving a best-case hit ratio of 8/24 = 33%.  Setting
        # group_size=48 collapses back to M-6 (singleton).
        self.pim_layer_group_size = int(pim_layer_group_size)
        self.enable_speculative_preload_gptq = bool(enable_speculative_preload_gptq)
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
        # ADR-002 M-9: routing-locality diagnostic.
        #
        # Every call to `_submit_forward_real` sees the topk_ids that the
        # router picked for this layer's CPU-side experts.  We keep the
        # set of active (= any-token-routed-to-it) CPU-side experts from
        # the previous call and compare via Jaccard similarity:
        #
        #     locality_t = |prev ∩ curr| / |prev ∪ curr|
        #
        # locality_t = 1 means the exact same expert set as last step
        # (perfect slot-cache reuse).  locality_t = 0 means no overlap
        # (MRAM slot cache cannot help).  Summed and counted separately
        # for prefill vs decode because they have very different dynamics.
        #
        # Histogram bins cover [0, 1] in 10% increments so the aggregate
        # distribution is visible in offload_diagnostics without per-step
        # storage.
        self._prev_active_cpu_experts_forward: frozenset[int] | None = None
        self.locality_decode_jaccard_sum: float = 0.0
        self.locality_decode_jaccard_count: int = 0
        self.locality_prefill_jaccard_sum: float = 0.0
        self.locality_prefill_jaccard_count: int = 0
        # 11 bins: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0), [1.0-1.0]
        self.locality_decode_jaccard_histogram: list[int] = [0] * 11

        # ADR-002 M-10: async PIM submit.
        #
        # When ``enable_async_pim_submit`` is True, ``submit_forward``
        # does not block on the DPU — it spawns a background thread
        # that runs ``_submit_forward_real`` while ``HybridMoE.forward``
        # continues with GPU attention / GPU-resident experts.
        # ``sync_forward`` then joins the thread before reading back
        # ``_fallback_output``.
        #
        # Python's GIL is released inside ctypes calls (dpu_launch,
        # dpu_push_xfer, memcpy bindings), so the PIM thread really
        # runs concurrently with the GPU side — no C-level threading
        # is required.  Default True because the worst case is "same
        # latency as sync path minus the thread spawn overhead (~30us)"
        # and the best case is 100 ms+/tok savings when GPU attention
        # is substantial.
        self.enable_async_pim_submit: bool = bool(enable_async_pim_submit)
        # ADR-002 M-24 Stage B: C-level fused gate_up + silu*up + down.
        # When True, `_run_quantized_experts_batched_on_dpu` dispatches
        # to the single-ctypes-call fused path (`_run_quantized_experts_c_fused`)
        # which collapses the Python-side two-RT + silu*up middleman into
        # one C function, reducing per-layer Python↔C overhead.
        # Falls back to the legacy batched path on any failure.  Default
        # off to keep the M-23.1 production path bit-identical.
        self.enable_c_fused_kernel: bool = bool(enable_c_fused_kernel)
        # ADR-002 M-24 Stage A: reserved for upcoming C-level async submit.
        # Wired here so __init__ signatures stay stable once Stage A lands;
        # currently a no-op because the submit path still goes through
        # the Python (or M-10) async infrastructure.
        self.enable_c_async_submit: bool = bool(enable_c_async_submit)
        # ADR-002 M-25 Stage A: replace blocking ``.to("cpu")`` in the
        # decode hot path with pinned + non_blocking copies so the GPU
        # expert loop does not serialise behind every submit_forward.
        # Only active when CUDA is actually available and the caller
        # goes through ``_submit_forward_c_async`` (the M-24 Stage A
        # path); sync real path still uses ``.to("cpu")`` directly
        # because it doesn't interleave with GPU work anyway.
        self.enable_m25_pinned_d2h: bool = bool(enable_m25_pinned_d2h)
        # ADR-002 M-26: move the Python body of _submit_forward_c_async
        # (expert loop, preload, ctypes array construction, C async
        # submit) onto a per-layer background Python thread.  The main
        # thread only runs the D2H pinned copy + cuda_stream sync (M-25
        # invariant), then spawns the thread and returns, letting the
        # HybridMoE GPU expert loop start ~300-500us sooner per layer.
        #
        # Correctness contract:
        #   - ctypes calls (preload, submit_async) auto-release the GIL,
        #     so the background thread mostly does not compete with the
        #     main GPU kernel launches.
        #   - sync_forward joins the spawn thread first, then waits on
        #     the handle as before.  Numerical output is bit-identical.
        #   - PIM still performs 100% of offloaded-expert compute; this
        #     is pure host-side orchestration.
        self.enable_m26_threaded_submit: bool = bool(enable_m26_threaded_submit)
        # Counters for the M-26 threaded-submit path (diagnostics only).
        self.m26_threaded_submit_count: int = 0
        self.m26_threaded_submit_wait_sum: float = 0.0
        self.m26_threaded_submit_wait_count: int = 0
        self.m26_threaded_submit_exc: Optional[BaseException] = None
        self._m26_submit_thread: Optional[threading.Thread] = None
        # Lazy pinned-buffer cache keyed by (batch_size, hidden_size,
        # top_k).  Each slot holds three pinned tensors sized for
        # (flat_cpu, topk_ids_cpu, topk_weights_cpu).  We only allocate
        # on first use to avoid paying pin cost on backends that never
        # hit the M-25 path.
        self._m25_pinned_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # M-25 Stage B: pinned staging buffer for sync_forward_c_async's
        # H2D of the per-layer output.  Keyed by shape tuple.
        self._m25_output_pinned_cache: dict[tuple[int, ...], torch.Tensor] = {}
        # Counters for the C fused path (M-24 Stage B diagnostics).
        self.c_fused_calls: int = 0
        self.c_fused_experts_processed: int = 0
        self.c_fused_fallback_count: int = 0
        # Stage A C-async state.  A submit stashes the handle + per-
        # expert metadata; the matching sync_forward joins the handle
        # and fills the output via index_add_.  Strictly one in-flight
        # job per PIMMoEBackend instance (per-layer).
        self._c_async_handle: Any = None
        self._c_async_meta: Optional[dict[str, Any]] = None
        self.c_async_submit_count: int = 0
        self.c_async_fallback_count: int = 0
        self.c_async_sync_wait_seconds_sum: float = 0.0
        self.c_async_sync_wait_seconds_count: int = 0
        self._async_thread: Optional[threading.Thread] = None
        self._async_exc: Optional[BaseException] = None
        # Latency telemetry (decode only).  submit_to_sync_wait_seconds
        # is the host-side wait the caller paid at sync time; if this
        # number is close to zero on average, async submit is hiding
        # all of the DPU work behind GPU work (the goal).
        self.async_submit_count: int = 0
        self.async_sync_wait_seconds_sum: float = 0.0
        self.async_sync_wait_seconds_count: int = 0
        self.runtime = None
        self.expert_runtime = None
        self.quantized_runtime: Optional[PIMQuantizedRuntime] = None
        # ADR-002 M-5: second quantized runtime pinned to the down
        # projection so that an expert's gate+up bundle and down
        # bundle can coexist in separate DPU MRAM pools.  Defaults
        # to the same object as ``quantized_runtime`` if dual
        # allocation is unavailable.
        self.quantized_runtime_down: Optional[PIMQuantizedRuntime] = None
        # Counters the backend maintains itself so diagnostics don't
        # depend on the aggregate of two singleton runtimes.
        self.quantized_preload_hits_local: int = 0
        self.quantized_preload_misses_local: int = 0
        # ADR-002 M-13: per-layer deltas of PIMQuantizedRuntime native
        # profile counters.  Runtime objects are shared across layers,
        # so diagnostics must aggregate before/after deltas locally.
        self.quantized_profile_load_count_local: int = 0
        self.quantized_profile_run_count_local: int = 0
        self.quantized_profile_seconds_sum_local: dict[str, float] = {
            field: 0.0
            for field in (
                *PIMQuantizedRuntime.PROFILE_LOAD_FIELDS,
                *PIMQuantizedRuntime.PROFILE_RUN_FIELDS,
            )
        }
        self.quantized_batched_expert_groups_local: int = 0
        self.quantized_batched_experts_local: int = 0
        # ADR-002 M-17.3: counts how many batched groups successfully
        # took the down-preload-before-gate-up-infer overlap path.
        self.quantized_down_preload_overlap_local: int = 0
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
                # Dual-runtime path (M-5): separate rank pools for
                # gate_up bundle vs down.  Falls back gracefully to the
                # single-runtime path if allocation fails.
                gate_up_rt, down_rt = self._try_init_quantized_runtimes_dual()
                if gate_up_rt is not None:
                    self.quantized_runtime = gate_up_rt
                    self.quantized_runtime_down = down_rt
                else:
                    self.quantized_runtime = self._try_init_quantized_runtime()
                    self.quantized_runtime_down = self.quantized_runtime
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
        """Legacy single-slot quantized runtime.

        Retained for backwards-compatibility paths that don't want the
        dual-runtime split introduced in M-5.  New code should prefer
        :meth:`_try_init_quantized_runtimes` below, which returns two
        independent runtimes that let each expert's gate+up bundle and
        down projection stay resident on separate DPU rank pools at
        the same time.
        """
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

    def _try_init_quantized_runtimes_dual(
        self,
    ) -> tuple[Optional[PIMQuantizedRuntime], Optional[PIMQuantizedRuntime]]:
        """
        ADR-002 M-5: allocate two independent PIMQuantizedRuntime
        instances — one for the fused gate+up bundle, one for the
        down projection.  The two runtimes back onto *different* DPU
        rank pools (keyed by ``profile`` in ``get_shared``), so a
        single expert's gate_up preload no longer evicts its own
        down preload (and vice versa).

        This removes one preload miss per expert per decode step.
        M-4 diagnostics show preload_hit_ratio = 0% which burns
        ~1.45 ms/call × 14.68 calls/layer = ~21 ms/layer/step in
        host->DPU weight transfer — the biggest single cost the
        cost-model routing couldn't see.

        ADR-002 M-7: the profile key now also embeds a
        ``layer_group_id`` so neighbouring layer groups each get their
        own 8-slot MRAM LRU instead of 48 layers beating on a single
        shared 8-slot cache.  The group size is controlled by
        :attr:`pim_layer_group_size` (default 3, giving 16 groups over
        48 layers — 16*2 runtimes = 32 DPU ranks, well under the 39
        visible on this machine).  If allocation fails for any group
        we gracefully fall back to the single-runtime path.
        """
        if not self.has_cpu_experts or self.visible_pim_ranks <= 0 or not self.is_gptq:
            return None, None

        # M-7: per-layer-group scoping of the slot table.
        group_size = max(1, int(self.pim_layer_group_size))
        group_id = self.layer_idx // group_size
        profile_suffix = f"g{group_id}" if group_size > 1 else f"l{self.layer_idx}"
        # ADR-002 M-8: ``instance_key`` is Python-cache-only; it must NOT
        # be passed through to UPMEM's dpu_alloc_ranks (M-7's original
        # code was fine only because the .so early-returned on the
        # second init).  Keep ``profile`` as what the user configured
        # (usually empty string) and use ``instance_key`` for the key.
        gate_up_key = (
            f"{self.pim_profile}|gate_up|{profile_suffix}"
            if profile_suffix
            else f"{self.pim_profile}|gate_up"
        )
        down_key = (
            f"{self.pim_profile}|down|{profile_suffix}"
            if profile_suffix
            else f"{self.pim_profile}|down"
        )

        try:
            gate_up_rt = PIMQuantizedRuntime.get_shared(
                profile=self.pim_profile,
                instance_key=gate_up_key,
                rank_count=max(1, self.pim_rank_count),
            )
        except Exception:
            self._record_fallback("quantized_runtime_gate_up_init_failed")
            gate_up_rt = None

        try:
            down_rt = PIMQuantizedRuntime.get_shared(
                profile=self.pim_profile,
                instance_key=down_key,
                rank_count=max(1, self.pim_rank_count),
            )
        except Exception:
            self._record_fallback("quantized_runtime_down_init_failed")
            down_rt = None

        # Fallback behaviour: if the dual allocation partially fails,
        # collapse to whichever runtime did come up so the code path
        # still works (at the cost of losing the M-5 win).
        if gate_up_rt is None and down_rt is None:
            return None, None
        if gate_up_rt is None:
            gate_up_rt = down_rt
        if down_rt is None:
            down_rt = gate_up_rt
        return gate_up_rt, down_rt

    # ── Expert ID for weight residency tracking ────────────────────────

    def _expert_id(self, cpu_slot: int) -> int:
        """Stable expert identity for DPU residency tracking."""
        return hash((self.layer_idx, cpu_slot)) & 0xFFFFFFFFFFFFFFFF

    def _snapshot_quantized_profile(
        self, runtime: Optional[PIMQuantizedRuntime]
    ) -> dict[str, float | int]:
        if runtime is None:
            return {}
        try:
            return runtime.profile_counters()
        except Exception:
            return {}

    def _accumulate_quantized_profile_delta(
        self,
        before: dict[str, float | int],
        after: dict[str, float | int],
    ) -> None:
        if not after:
            return
        self.quantized_profile_load_count_local += max(
            0, int(after.get("load_count", 0) or 0) - int(before.get("load_count", 0) or 0)
        )
        self.quantized_profile_run_count_local += max(
            0, int(after.get("run_count", 0) or 0) - int(before.get("run_count", 0) or 0)
        )
        for field in self.quantized_profile_seconds_sum_local:
            key = f"{field}_sum"
            delta = float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
            if delta > 0.0:
                self.quantized_profile_seconds_sum_local[field] += delta

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

        ADR-002 M-5:  gate_up bundle and down bundle live on separate
        PIMQuantizedRuntime instances (backed by distinct DPU rank
        pools) so the gate_up preload no longer evicts this expert's
        own down preload.  This alone does not change miss count
        *within* a step (every new expert still misses on both bundles
        since MoE routing changes each step), but it prevents the
        pathological "same expert re-misses its own bundles twice per
        forward" pattern that dominated M-4's 1.1 M preload misses.
        It also makes speculative preload of the next expert's down
        bundle possible in future milestones.
        """
        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime
        if rt_gate_up is None:
            raise RuntimeError("PIM quantized runtime is not available.")

        gptq = self._gptq_experts.get(cpu_slot)
        if gptq is None:
            raise RuntimeError(f"No GPTQ weights for cpu_slot={cpu_slot}")

        kernel_mode = 4  # int8 fixed-point (best for batch=1 decode)

        base_eid = self._expert_id(cpu_slot)
        gate_up_eid = base_eid ^ 0x1212121212121212

        # Remember hit/miss counts before the call so we can derive a
        # per-backend-instance delta.  preload_hits/misses are runtime-
        # global, so aggregating at the backend needs a local delta.
        pre_hits = rt_gate_up.preload_hits
        pre_miss = rt_gate_up.preload_misses
        pre_profile = self._snapshot_quantized_profile(rt_gate_up)

        gate, up = rt_gate_up.preload_and_infer_concat(
            gate_up_eid,
            gptq["gate"],
            gptq["up"],
            states,
            kernel_mode=kernel_mode,
        )
        self.quantized_preload_hits_local += (rt_gate_up.preload_hits - pre_hits)
        self.quantized_preload_misses_local += (rt_gate_up.preload_misses - pre_miss)
        self._accumulate_quantized_profile_delta(
            pre_profile, self._snapshot_quantized_profile(rt_gate_up)
        )

        hidden = F.silu(gate) * up

        down_eid = base_eid ^ 0x3333333333333333
        pre_hits_d = rt_down.preload_hits
        pre_miss_d = rt_down.preload_misses
        pre_profile_d = self._snapshot_quantized_profile(rt_down)
        rt_down.preload(down_eid, gptq["down"], kernel_mode)
        output = rt_down.infer(hidden)
        self.quantized_preload_hits_local += (rt_down.preload_hits - pre_hits_d)
        self.quantized_preload_misses_local += (rt_down.preload_misses - pre_miss_d)
        self._accumulate_quantized_profile_delta(
            pre_profile_d, self._snapshot_quantized_profile(rt_down)
        )

        self.real_dpu_quantized_calls += 2
        self.real_dpu_expert_calls += 1
        self.last_kernel_cycles = rt_down.last_cycles()
        return output

    def _acquire_pinned_submit_buffers(
        self,
        *,
        batch_size: int,
        hidden_size: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ADR-002 M-25 Stage A: fetch (and lazily allocate) pinned
        CPU buffers for the submit_forward D2H pipeline.

        Returns a ``(flat_cpu, topk_ids_cpu, topk_weights_cpu)`` tuple
        dimensioned identically to the tensors that ``.to("cpu", ...)``
        would otherwise produce:
          * flat_cpu         [batch_size, hidden_size]  float32
          * topk_ids_cpu     [batch_size, top_k]        long
          * topk_weights_cpu [batch_size, top_k]        float32

        Buffers are reused across decode steps; pin cost is paid once
        per distinct (batch, hidden, top_k) shape.  For Qwen3 decode
        this cache only ever holds a single entry (batch=1).
        """
        key = (int(batch_size), int(hidden_size), int(top_k))
        cached = self._m25_pinned_cache.get(key)
        if cached is not None:
            return cached
        flat_cpu = torch.empty(
            batch_size, hidden_size, dtype=torch.float32, pin_memory=True
        )
        topk_ids_cpu = torch.empty(
            batch_size, top_k, dtype=torch.long, pin_memory=True
        )
        topk_weights_cpu = torch.empty(
            batch_size, top_k, dtype=torch.float32, pin_memory=True
        )
        buffers = (flat_cpu, topk_ids_cpu, topk_weights_cpu)
        self._m25_pinned_cache[key] = buffers
        return buffers

    def _acquire_pinned_output_buffer(self, shape: torch.Size) -> torch.Tensor:
        """M-25 Stage B pinned staging buffer for the output H2D path.

        Returns a float32 pinned tensor with the given shape; cached
        across calls keyed by the shape tuple.
        """
        key = tuple(int(d) for d in shape)
        cached = self._m25_output_pinned_cache.get(key)
        if cached is not None:
            return cached
        buffer = torch.empty(shape, dtype=torch.float32, pin_memory=True)
        self._m25_output_pinned_cache[key] = buffer
        return buffer

    def _submit_forward_c_async(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> bool:
        """ADR-002 M-24 Stage A: submit the fused op to a C pthread worker.

        Returns True on successful submission (handle stashed in
        ``self._c_async_handle`` + metadata in ``self._c_async_meta``).
        Returns False when the path is unavailable (caller should fall
        back to the synchronous real path).

        PIM still performs the gate_up and down matvecs — this is purely
        an orchestration-overlap change so that ``HybridMoE.forward``'s
        subsequent GPU expert loop runs concurrently with DPU work.
        """
        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime
        if rt_gate_up is None or rt_down is None:
            return False

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        cpu_mask = ~self.gpu_experts_mask.bool()

        # ADR-002 M-25 Stage A: pinned-buffer + non_blocking D2H.
        #
        # Before M-25 these three copies were plain `.to("cpu", ...)`
        # which are synchronous D2H transfers — they block Python
        # until the CUDA queue drains, serialising the subsequent GPU
        # expert loop (92 experts × 48 layers × 32 tokens) behind
        # every submit_forward entry.  Profiling (ADR-002 §35) showed
        # ~11s of the 26.82s M-24 A decode came from this serialisation.
        #
        # We now:
        #   1. Allocate (once per batch size) pinned host buffers that
        #      hold the D2H shadows (flat / topk_ids / topk_weights).
        #   2. Issue ``copy_(..., non_blocking=True)`` so CUDA queues
        #      the copy on the caller's stream and Python returns
        #      immediately.  The HybridMoE.forward GPU expert loop
        #      can now overlap fully with the D2H transfer + PIM work.
        #   3. Sync the cuda_stream exactly once before handing the
        #      pinned buffers to the PIM C worker.  This is the only
        #      place Python blocks on GPU work in the hot path; M-24
        #      Stage A's c_async_wait already covers PIM completion.
        #
        # Falls back to the pre-M-25 synchronous `.to("cpu")` path when
        # CUDA is unavailable (test harness) or hidden_states is already
        # on CPU.
        use_pinned_path = (
            hidden_states.is_cuda
            and torch.cuda.is_available()
            and self.enable_m25_pinned_d2h
        )
        if use_pinned_path:
            flat_cpu, topk_ids_cpu, topk_weights_cpu = self._acquire_pinned_submit_buffers(
                batch_size=batch_size,
                hidden_size=flat.shape[-1],
                top_k=topk_ids.shape[-1],
            )
            flat_cpu.copy_(flat.to(dtype=torch.float32), non_blocking=True)
            topk_ids_cpu.copy_(topk_ids.to(dtype=torch.long), non_blocking=True)
            topk_weights_cpu.copy_(topk_weights.to(dtype=torch.float32), non_blocking=True)
            # Only now do we need the data on the host side for Python-
            # driven preload + request assembly below.  Sync the default
            # CUDA stream exactly once; HybridMoE.forward already holds
            # the stream handle used by the submit/sync pair.
            torch.cuda.current_stream().synchronize()
        else:
            flat_cpu = flat.to("cpu", dtype=torch.float32)
            topk_ids_cpu = topk_ids.to("cpu", dtype=torch.long)
            topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)

        # ADR-002 M-25 Stage A: diagnostic counters now computed on
        # CPU-materialised tensors (free — the D2H already happened
        # above, and .item() on a CPU tensor is not a CUDA sync).
        # Preserves pim_compute_participation_ratio semantics.
        routed_cpu = cpu_mask[topk_ids_cpu]
        self.offloaded_pairs += int(routed_cpu.sum().item())
        self.offloaded_tokens += int(routed_cpu.any(dim=1).sum().item())

        # ADR-002 M-26: optionally offload the remaining Python work
        # (expert loop + preload + ctypes submit) to a background
        # thread so the main thread returns to HybridMoE.forward ~300-
        # 500us sooner per layer.  All ctypes calls in the offloaded
        # section auto-release the GIL, so the main thread's GPU
        # expert loop is not meaningfully blocked.
        if self.enable_m26_threaded_submit:
            # Clear any stale thread state before spawning (defensive;
            # sync_forward should have joined previous layer's thread).
            if self._m26_submit_thread is not None and self._m26_submit_thread.is_alive():
                self._m26_submit_thread.join()
            self.m26_threaded_submit_exc = None
            hs_device = hidden_states.device
            hs_dtype = hidden_states.dtype
            hs_shape = hidden_states.shape

            def _worker() -> None:
                try:
                    self._do_c_async_submit_work(
                        flat_cpu=flat_cpu,
                        topk_ids_cpu=topk_ids_cpu,
                        topk_weights_cpu=topk_weights_cpu,
                        cpu_mask=cpu_mask,
                        rt_gate_up=rt_gate_up,
                        rt_down=rt_down,
                        batch_size=batch_size,
                        hs_device=hs_device,
                        hs_dtype=hs_dtype,
                        hs_shape=hs_shape,
                        hidden_states=hidden_states,
                    )
                except BaseException as exc:  # noqa: BLE001
                    self.m26_threaded_submit_exc = exc

            t = threading.Thread(
                target=_worker,
                name=f"pim_m26_submit_L{self.layer_idx}",
                daemon=True,
            )
            t.start()
            self._m26_submit_thread = t
            self.m26_threaded_submit_count += 1
            return True

        # Non-threaded (M-25) path: run the work inline on the main thread.
        return self._do_c_async_submit_work(
            flat_cpu=flat_cpu,
            topk_ids_cpu=topk_ids_cpu,
            topk_weights_cpu=topk_weights_cpu,
            cpu_mask=cpu_mask,
            rt_gate_up=rt_gate_up,
            rt_down=rt_down,
            batch_size=batch_size,
            hs_device=hidden_states.device,
            hs_dtype=hidden_states.dtype,
            hs_shape=hidden_states.shape,
            hidden_states=hidden_states,
        )

    def _do_c_async_submit_work(
        self,
        *,
        flat_cpu: torch.Tensor,
        topk_ids_cpu: torch.Tensor,
        topk_weights_cpu: torch.Tensor,
        cpu_mask: torch.Tensor,
        rt_gate_up: PIMQuantizedRuntime,
        rt_down: PIMQuantizedRuntime,
        batch_size: int,
        hs_device: torch.device,
        hs_dtype: torch.dtype,
        hs_shape: torch.Size,
        hidden_states: torch.Tensor,
    ) -> bool:
        """ADR-002 M-26: the Python body of ``_submit_forward_c_async``
        after the (main-thread-only) pinned D2H + cuda sync.

        Extracted so it can run either inline on the main thread
        (legacy M-24/M-25 path) or on a per-layer background Python
        thread (M-26).  All ctypes calls here auto-release the GIL, so
        when run in a background thread the main thread's GPU expert
        loop continues almost uncontested.
        """
        # Collect activated CPU experts.  Same logic as _submit_forward_real.
        activated_cpu_experts: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
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
            if cpu_slot not in self._gptq_experts:
                # Fused path requires GPTQ weights for every activated expert.
                return False
            activated_cpu_experts.append(
                (expert_idx, cpu_slot, token_indices, match)
            )

        if not activated_cpu_experts:
            # No PIM work to submit; mark output as zeros so sync_forward
            # has something to return.  Allocate on the caller-requested
            # device/dtype so both threaded and inline paths behave the
            # same (hidden_states is only used for shape metadata here —
            # it is NOT touched from the background thread for data).
            self._fallback_output = torch.zeros(
                hs_shape, dtype=hs_dtype, device=hs_device
            )
            return True

        kernel_mode = 4
        # Preload every expert's bundles + build the request list, same
        # as the sync fused path.
        requests: list[
            tuple[torch.Tensor, int, int, int, int, int, int, int, int]
        ] = []
        per_expert_records: list[dict[str, Any]] = []
        pre_hits_gu = rt_gate_up.preload_hits
        pre_miss_gu = rt_gate_up.preload_misses
        pre_profile_gu = self._snapshot_quantized_profile(rt_gate_up)
        pre_hits_dn = rt_down.preload_hits
        pre_miss_dn = rt_down.preload_misses
        pre_profile_dn = self._snapshot_quantized_profile(rt_down)
        for expert_idx, cpu_slot, token_indices, match in activated_cpu_experts:
            gptq = self._gptq_experts.get(cpu_slot)
            if gptq is None:
                return False
            states = flat_cpu[token_indices]
            base_eid = self._expert_id(cpu_slot)
            gate_up_eid = base_eid ^ 0x1212121212121212
            down_eid = base_eid ^ 0x3333333333333333
            gu_slot, gu_padded_in, gu_concat, gate_cols, up_cols = \
                rt_gate_up.preload_concat_and_get_slot(
                    gate_up_eid, gptq["gate"], gptq["up"], kernel_mode=kernel_mode,
                )
            dn_slot, dn_padded_in, dn_padded_out, dn_orig_out = \
                rt_down.preload_and_get_slot(
                    down_eid, gptq["down"], kernel_mode,
                )
            if gate_cols != up_cols or dn_padded_in != up_cols:
                return False
            requests.append(
                (states, gu_slot, gu_padded_in, gu_concat,
                 gate_cols, up_cols, dn_slot, dn_padded_in, dn_padded_out)
            )
            per_expert_records.append(
                {
                    "token_indices": token_indices,
                    "match": match,
                    "dn_orig_out": dn_orig_out,
                }
            )

        handle = PIMQuantizedRuntime.submit_many_fused_silu_async(
            rt_gate_up, rt_down, requests,
        )
        self._c_async_handle = handle
        self._c_async_meta = {
            "per_expert_records": per_expert_records,
            "topk_weights_cpu": topk_weights_cpu,
            "batch_size": batch_size,
            "hidden_size": self.hidden_size,
            "device": hs_device,
            "hidden_dtype": hs_dtype,
            "hidden_shape": hs_shape,
            "rt_gate_up": rt_gate_up,
            "rt_down": rt_down,
            "pre_hits_gu": pre_hits_gu,
            "pre_miss_gu": pre_miss_gu,
            "pre_profile_gu": pre_profile_gu,
            "pre_hits_dn": pre_hits_dn,
            "pre_miss_dn": pre_miss_dn,
            "pre_profile_dn": pre_profile_dn,
            "n": len(per_expert_records),
        }
        self.c_async_submit_count += 1
        return True

    def _sync_forward_c_async(self) -> Optional[torch.Tensor]:
        """Join the C pthread worker started by ``_submit_forward_c_async``
        and assemble the per-layer output tensor.

        Returns the fallback output tensor (GPU/CPU dtype-matching), or
        None if no C-async handle is pending (caller falls back to the
        super().sync_forward path).
        """
        handle = self._c_async_handle
        meta = self._c_async_meta
        if handle is None or meta is None:
            return None
        self._c_async_handle = None
        self._c_async_meta = None

        import time as _time
        wait_start = _time.perf_counter()
        try:
            down_outputs = handle.wait()
        except Exception:
            self._record_fallback("c_async_wait_failed")
            self.c_async_fallback_count += 1
            # Re-raise so sync_forward surfaces the error to the caller.
            raise
        wait_s = _time.perf_counter() - wait_start
        self.c_async_sync_wait_seconds_sum += wait_s
        self.c_async_sync_wait_seconds_count += 1

        batch_size = meta["batch_size"]
        hidden_size = meta["hidden_size"]
        device = meta["device"]
        hidden_dtype = meta["hidden_dtype"]
        per_expert_records = meta["per_expert_records"]
        topk_weights_cpu = meta["topk_weights_cpu"]

        output = torch.zeros(
            batch_size, hidden_size, dtype=torch.float32, device="cpu"
        )
        for rec, down_output in zip(per_expert_records, down_outputs):
            dn_orig_out = rec["dn_orig_out"]
            expert_output = down_output[:, :dn_orig_out].contiguous()
            token_indices = rec["token_indices"]
            match = rec["match"]
            row_idx, col_idx = torch.where(match[token_indices])
            weights = (
                topk_weights_cpu[token_indices[row_idx], col_idx]
                .to(dtype=expert_output.dtype)
                .unsqueeze(1)
            )
            output.index_add_(
                0, token_indices[row_idx], expert_output[row_idx] * weights
            )

        # Profile accounting (deferred from submit to wait).
        rt_gate_up = meta["rt_gate_up"]
        rt_down = meta["rt_down"]
        self.quantized_preload_hits_local += (
            rt_gate_up.preload_hits - meta["pre_hits_gu"]
        )
        self.quantized_preload_misses_local += (
            rt_gate_up.preload_misses - meta["pre_miss_gu"]
        )
        self._accumulate_quantized_profile_delta(
            meta["pre_profile_gu"],
            self._snapshot_quantized_profile(rt_gate_up),
        )
        if rt_down is not rt_gate_up:
            self.quantized_preload_hits_local += (
                rt_down.preload_hits - meta["pre_hits_dn"]
            )
            self.quantized_preload_misses_local += (
                rt_down.preload_misses - meta["pre_miss_dn"]
            )
            self._accumulate_quantized_profile_delta(
                meta["pre_profile_dn"],
                self._snapshot_quantized_profile(rt_down),
            )

        n = meta["n"]
        self.real_dpu_quantized_calls += 2 * n
        self.real_dpu_expert_calls += n
        self.last_kernel_cycles = rt_down.last_cycles()
        self.quantized_batched_expert_groups_local += 1
        self.quantized_batched_experts_local += n
        # Treat the C async path as a fused-call variant for counter parity.
        self.c_fused_calls += 1
        self.c_fused_experts_processed += n

        hidden_shape = meta["hidden_shape"]
        # ADR-002 M-25 Stage B: non_blocking H2D of the layer output.
        #
        # The old code did ``output.view(hidden_shape).to(device, dtype)``
        # which on CUDA is a blocking H2D — it serialises the final
        # add in HybridMoE.forward (``final_gpu_states + cpu_output``)
        # behind a host-side wait.  By pinning ``output`` (allocated
        # as pin_memory above — see _m25_output_buffer_cache) and
        # using non_blocking=True, the copy is queued on the current
        # CUDA stream and returns immediately; the subsequent GPU add
        # on the same stream will naturally wait for the copy to
        # complete, with no Python sync.
        if (
            self.enable_m25_pinned_d2h
            and torch.cuda.is_available()
            and isinstance(device, torch.device)
            and device.type == "cuda"
        ):
            # Use a pinned staging buffer that matches `output`'s shape.
            staging = self._acquire_pinned_output_buffer(output.shape)
            staging.copy_(output)  # host-to-host, fast on pinned dest
            result = torch.empty(
                output.shape, dtype=hidden_dtype, device=device
            )
            result.copy_(staging.to(dtype=hidden_dtype), non_blocking=True)
            return result.view(hidden_shape)
        return output.view(hidden_shape).to(device=device, dtype=hidden_dtype)

    def _run_quantized_experts_c_fused(
        self,
        activated_cpu_experts: list[tuple[int, int, torch.Tensor, torch.Tensor]],
        flat_cpu: torch.Tensor,
        topk_weights_cpu: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        """ADR-002 M-24 Stage B: single-ctypes-call fused gate_up + silu*up + down.

        PIM still performs all the matvec work (this method never bypasses
        the DPU).  The only orchestration change vs
        ``_run_quantized_experts_batched_on_dpu``:

          * Two ``infer_many_raw`` ctypes round-trips → one
            ``infer_many_fused_silu``.
          * ``F.silu(gate) * up`` in Python (+ ``.contiguous()`` slices)
            → tight fp32 loop in C (see
            ``host_quantized_bridge.c::pim_quantized_run_many_fused_silu``).

        Returns True on success (``output`` is populated via index_add_),
        False when the path is unavailable for this batch (caller should
        fall back to the legacy per-phase path).
        """
        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime
        if rt_gate_up is None or rt_down is None:
            return False
        if not activated_cpu_experts:
            return False

        kernel_mode = 4

        # Phase A (Python): preload gate_up concat bundle + down bundle
        # for every activated expert, collecting the per-expert
        # (slot, padded shape) tuples needed to construct the fused C
        # request.  Numerically identical to the legacy path's preload
        # sequence.
        requests: list[
            tuple[torch.Tensor, int, int, int, int, int, int, int, int]
        ] = []
        per_expert_records: list[dict[str, Any]] = []
        pre_hits_gu = rt_gate_up.preload_hits
        pre_miss_gu = rt_gate_up.preload_misses
        pre_profile_gu = self._snapshot_quantized_profile(rt_gate_up)
        pre_hits_dn = rt_down.preload_hits
        pre_miss_dn = rt_down.preload_misses
        pre_profile_dn = self._snapshot_quantized_profile(rt_down)
        for expert_idx, cpu_slot, token_indices, match in activated_cpu_experts:
            gptq = self._gptq_experts.get(cpu_slot)
            if gptq is None:
                return False
            states = flat_cpu[token_indices]
            base_eid = self._expert_id(cpu_slot)
            gate_up_eid = base_eid ^ 0x1212121212121212
            down_eid = base_eid ^ 0x3333333333333333

            gu_slot, gu_padded_in, gu_concat_rows, gate_cols, up_cols = \
                rt_gate_up.preload_concat_and_get_slot(
                    gate_up_eid,
                    gptq["gate"],
                    gptq["up"],
                    kernel_mode=kernel_mode,
                )
            dn_slot, dn_padded_in, dn_padded_out, dn_orig_out = \
                rt_down.preload_and_get_slot(
                    down_eid,
                    gptq["down"],
                    kernel_mode,
                )
            # For SwiGLU gate_cols == up_cols (they are the two halves of
            # the concat).  The fused C API enforces this; also guard here.
            if gate_cols != up_cols:
                return False
            # The down input dim must equal up_cols (SwiGLU intermediate
            # width).  If not, caller is using a non-Qwen3 shape — bail.
            if dn_padded_in != up_cols:
                return False

            requests.append(
                (
                    states,
                    gu_slot,
                    gu_padded_in,
                    gu_concat_rows,
                    gate_cols,
                    up_cols,
                    dn_slot,
                    dn_padded_in,
                    dn_padded_out,
                )
            )
            per_expert_records.append(
                {
                    "token_indices": token_indices,
                    "match": match,
                    "dn_orig_out": dn_orig_out,
                }
            )

        # Phase B (C): single fused call.  The caller has already paid
        # the preload DMAs above; this call does only input encode +
        # gate_up launch + silu*up + down launch + output dequant.
        down_outputs = PIMQuantizedRuntime.infer_many_fused_silu(
            rt_gate_up,
            rt_down,
            requests,
        )

        # Phase C (Python): scatter per-expert down outputs into the
        # layer's running output buffer using the routing weights, exactly
        # as the legacy path does.  Same tensor shapes + dtypes, same
        # index_add_ semantics.
        for req_idx, (rec, down_output) in enumerate(zip(per_expert_records, down_outputs)):
            dn_orig_out = rec["dn_orig_out"]
            expert_output = down_output[:, :dn_orig_out].contiguous()
            token_indices = rec["token_indices"]
            match = rec["match"]
            row_idx, col_idx = torch.where(match[token_indices])
            weights = (
                topk_weights_cpu[token_indices[row_idx], col_idx]
                .to(dtype=expert_output.dtype)
                .unsqueeze(1)
            )
            output.index_add_(
                0, token_indices[row_idx], expert_output[row_idx] * weights
            )

        # Accounting: the fused path does the same number of DPU calls
        # as the legacy path (2 per expert: gate_up + down), preload
        # hits/misses via the same preload APIs, and the C kernel_cycles
        # are still stamped on the down ctx (fused C calls run_many on
        # down last).
        self.quantized_preload_hits_local += (rt_gate_up.preload_hits - pre_hits_gu)
        self.quantized_preload_misses_local += (rt_gate_up.preload_misses - pre_miss_gu)
        self._accumulate_quantized_profile_delta(
            pre_profile_gu, self._snapshot_quantized_profile(rt_gate_up)
        )
        if rt_down is not rt_gate_up:
            self.quantized_preload_hits_local += (rt_down.preload_hits - pre_hits_dn)
            self.quantized_preload_misses_local += (rt_down.preload_misses - pre_miss_dn)
            self._accumulate_quantized_profile_delta(
                pre_profile_dn, self._snapshot_quantized_profile(rt_down)
            )

        n = len(per_expert_records)
        self.real_dpu_quantized_calls += 2 * n
        self.real_dpu_expert_calls += n
        self.last_kernel_cycles = rt_down.last_cycles()
        self.quantized_batched_expert_groups_local += 1
        self.quantized_batched_experts_local += n
        self.c_fused_calls += 1
        self.c_fused_experts_processed += n
        return True

    def _run_quantized_experts_batched_on_dpu(
        self,
        activated_cpu_experts: list[tuple[int, int, torch.Tensor, torch.Tensor]],
        flat_cpu: torch.Tensor,
        topk_weights_cpu: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime
        if rt_gate_up is None or rt_down is None:
            return False
        if not activated_cpu_experts:
            return False

        # ADR-002 M-24 Stage B: if the C-fused kernel is enabled, try it
        # first.  It collapses the two batched DPU launches plus the
        # Python silu*up middleman into one C ctypes call, cutting per-
        # layer Python↔C overhead.  On any failure we fall back to the
        # legacy two-call path below so correctness is never at risk.
        if self.enable_c_fused_kernel:
            try:
                ok = self._run_quantized_experts_c_fused(
                    activated_cpu_experts,
                    flat_cpu,
                    topk_weights_cpu,
                    output,
                )
                if ok:
                    return True
                # ok=False means fused path explicitly bailed — count and
                # continue to legacy path below.
                self.c_fused_fallback_count += 1
            except Exception:
                self._record_fallback("c_fused_path_failed")
                self.c_fused_fallback_count += 1
                # fall through to legacy path

        kernel_mode = 4
        gate_entries: list[dict[str, Any]] = []
        pre_hits = rt_gate_up.preload_hits
        pre_miss = rt_gate_up.preload_misses
        pre_profile = self._snapshot_quantized_profile(rt_gate_up)
        for expert_idx, cpu_slot, token_indices, match in activated_cpu_experts:
            gptq = self._gptq_experts.get(cpu_slot)
            if gptq is None:
                return False
            states = flat_cpu[token_indices]
            base_eid = self._expert_id(cpu_slot)
            gate_up_eid = base_eid ^ 0x1212121212121212
            slot, padded_in, concat_rows, lhs_orig, rhs_orig = rt_gate_up.preload_concat_and_get_slot(
                gate_up_eid,
                gptq["gate"],
                gptq["up"],
                kernel_mode=kernel_mode,
            )
            gate_entries.append({
                "expert_idx": expert_idx,
                "cpu_slot": cpu_slot,
                "token_indices": token_indices,
                "match": match,
                "states": states,
                "slot": slot,
                "padded_in": padded_in,
                "concat_rows": concat_rows,
                "lhs_orig": lhs_orig,
                "rhs_orig": rhs_orig,
                "base_eid": base_eid,
                "gptq": gptq,
            })

        # ADR-002 M-17.3: when down lives on a DISTINCT DPU rank pool
        # (M-5 dual-runtime path), pre-issue all down preloads BEFORE
        # gate+up infer.  Their lut/qweight/scale ASYNC pushes (M-17.2)
        # then run on the down dpu_set in parallel with the gate+up
        # launch on the gate_up dpu_set, hiding most of the down weight
        # DMA behind gate+up compute and the host-side silu(gate)*up.
        #
        # When the two runtimes collapse to the same ctx (single-ctx
        # fallback), pre-issuing down preload would let down evict
        # slots that gate+up infer is about to read — keep the legacy
        # ordering in that case.
        down_preload_overlap = rt_down is not rt_gate_up
        if down_preload_overlap:
            pre_hits_d = rt_down.preload_hits
            pre_miss_d = rt_down.preload_misses
            pre_profile_d = self._snapshot_quantized_profile(rt_down)
            down_preload_records: list[tuple[int, int, int, int]] = []
            for entry in gate_entries:
                down_eid = entry["base_eid"] ^ 0x3333333333333333
                slot, padded_in, padded_out, orig_out = rt_down.preload_and_get_slot(
                    down_eid,
                    entry["gptq"]["down"],
                    kernel_mode,
                )
                down_preload_records.append((slot, padded_in, padded_out, orig_out))

        gate_outputs = rt_gate_up.infer_many_raw([
            (entry["states"], entry["slot"], entry["padded_in"], entry["concat_rows"])
            for entry in gate_entries
        ])
        self.quantized_preload_hits_local += (rt_gate_up.preload_hits - pre_hits)
        self.quantized_preload_misses_local += (rt_gate_up.preload_misses - pre_miss)
        self._accumulate_quantized_profile_delta(
            pre_profile, self._snapshot_quantized_profile(rt_gate_up)
        )

        down_entries: list[dict[str, Any]] = []
        if not down_preload_overlap:
            pre_hits_d = rt_down.preload_hits
            pre_miss_d = rt_down.preload_misses
            pre_profile_d = self._snapshot_quantized_profile(rt_down)
        for i, (entry, gate_up_output) in enumerate(zip(gate_entries, gate_outputs)):
            gate = gate_up_output[:, :entry["lhs_orig"]].contiguous()
            up = gate_up_output[:, entry["lhs_orig"] : entry["lhs_orig"] + entry["rhs_orig"]].contiguous()
            hidden = F.silu(gate) * up
            if down_preload_overlap:
                slot, padded_in, padded_out, orig_out = down_preload_records[i]
            else:
                down_eid = entry["base_eid"] ^ 0x3333333333333333
                slot, padded_in, padded_out, orig_out = rt_down.preload_and_get_slot(
                    down_eid,
                    entry["gptq"]["down"],
                    kernel_mode,
                )
            down_entries.append({
                **entry,
                "hidden": hidden,
                "down_slot": slot,
                "down_padded_in": padded_in,
                "down_padded_out": padded_out,
                "down_orig_out": orig_out,
            })

        down_outputs = rt_down.infer_many_raw([
            (entry["hidden"], entry["down_slot"], entry["down_padded_in"], entry["down_padded_out"])
            for entry in down_entries
        ])
        self.quantized_preload_hits_local += (rt_down.preload_hits - pre_hits_d)
        self.quantized_preload_misses_local += (rt_down.preload_misses - pre_miss_d)
        self._accumulate_quantized_profile_delta(
            pre_profile_d, self._snapshot_quantized_profile(rt_down)
        )

        for entry, down_output in zip(down_entries, down_outputs):
            expert_output = down_output[:, :entry["down_orig_out"]].contiguous()
            token_indices = entry["token_indices"]
            match = entry["match"]
            row_idx, col_idx = torch.where(match[token_indices])
            weights = topk_weights_cpu[token_indices[row_idx], col_idx].to(dtype=expert_output.dtype).unsqueeze(1)
            output.index_add_(0, token_indices[row_idx], expert_output[row_idx] * weights)

        n = len(down_entries)
        self.real_dpu_quantized_calls += 2 * n
        self.real_dpu_expert_calls += n
        self.last_kernel_cycles = rt_down.last_cycles()
        self.quantized_batched_expert_groups_local += 1
        self.quantized_batched_experts_local += n
        # ADR-002 M-17.3: expose whether the overlap path was taken so
        # diagnostics can show it landed at runtime.
        if down_preload_overlap:
            self.quantized_down_preload_overlap_local += 1
        return True

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

    def _speculative_preload_gptq(self, topk_ids: torch.Tensor) -> None:
        """
        ADR-002 M-7: GPTQ-aware speculative preload.

        At prefill end, take the top-N hottest CPU-side experts (by
        route frequency in this prefill's topk_ids) and upload their
        fused gate+up bundle AND down bundle into the 8-slot MRAM
        LRU of the matching layer-group runtime pair.  This warms
        the slot cache so the first few decode steps can hit instead
        of cold-missing every expert.

        Hit-ratio arithmetic (see ADR-002 §15):
        * Each runtime pair holds 8 slots × ``pim_layer_group_size``
          layers worth of working set per step.  Preloading top-N
          where N = min(NUM_SLOTS, top_k * group_size) gives us the
          densest warm-up without immediately thrashing.
        * Per-layer we preload only this layer's top experts — the
          runtime pair's slot table is shared across the group, so
          every layer in the group contributes.
        """
        if not self.is_gptq or self.quantized_runtime is None:
            return
        if not self.enable_speculative_preload_gptq:
            return

        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime

        cpu_mask = ~self.gpu_experts_mask.bool()
        flat_ids = topk_ids.view(-1).cpu().long()
        counts = torch.bincount(flat_ids, minlength=self.num_experts)
        counts[~cpu_mask] = 0

        if int(counts.max()) == 0:
            return

        # Pick at most NUM_SLOTS hot experts — we don't want this layer
        # alone to fill all slots, because neighbouring layers in the
        # group will also want room.
        per_layer_cap = max(1, rt_gate_up.NUM_SLOTS // max(1, self.pim_layer_group_size))
        # Clamp to the number of non-zero-count experts available.
        nonzero = int((counts > 0).sum().item())
        top_n = min(per_layer_cap, nonzero)
        if top_n <= 0:
            return

        hot_experts = counts.topk(top_n).indices.tolist()

        kernel_mode = 4
        preloaded = 0
        for expert_idx in hot_experts:
            cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
            if cpu_slot is None or cpu_slot not in self._gptq_experts:
                continue
            gptq = self._gptq_experts[cpu_slot]
            base_eid = self._expert_id(cpu_slot)
            gate_up_eid = base_eid ^ 0x1212121212121212
            down_eid = base_eid ^ 0x3333333333333333
            try:
                # gate+up concat bundle: we pre-prepare the padded
                # tensors *and* allocate the MRAM slot; actual DPU DMA
                # happens here.  Cost ≈ 0.96 ms/bundle (M-5 micro-bench).
                slot, was_resident = rt_gate_up._allocate_slot(gate_up_eid)
                if not was_resident:
                    # Use preload_and_infer_concat's cache-prepare +
                    # DMA path without the infer tail.  We call into
                    # the lower-level _prepare_concat helper + direct
                    # ctypes load so no spurious DPU launch happens.
                    if gate_up_eid not in rt_gate_up._weight_cache or \
                            rt_gate_up._weight_cache[gate_up_eid][6] <= 0:
                        concat_qw, concat_sc, padded_in, concat_rows, lhs_o, rhs_o = \
                            rt_gate_up._prepare_concat_quantized_weights(
                                gptq["gate"], gptq["up"], kernel_mode
                            )
                        rt_gate_up._weight_cache[gate_up_eid] = (
                            concat_qw, concat_sc, padded_in, concat_rows,
                            gptq["gate"].group_size, kernel_mode, rhs_o,
                        )
                    concat_qw, concat_sc, padded_in, concat_rows, gsize, km, _ = \
                        rt_gate_up._weight_cache[gate_up_eid]
                    import ctypes as _c
                    err = _c.create_string_buffer(rt_gate_up.ERROR_BUFFER_SIZE)
                    rc = rt_gate_up._lib.pim_quantized_load_weights(
                        rt_gate_up._handle,
                        _c.c_uint32(padded_in),
                        _c.c_uint32(concat_rows),
                        _c.c_uint32(gsize),
                        _c.c_uint32(km),
                        _c.c_void_p(concat_qw.data_ptr()),
                        _c.c_void_p(concat_sc.data_ptr()),
                        _c.c_uint32(slot),
                        err,
                        len(err),
                    )
                    if rc != 0:
                        raise RuntimeError(err.value.decode("utf-8", errors="replace"))
                    rt_gate_up._record_load_profile()
                    rt_gate_up.preload_misses += 1
                    rt_gate_up._resident_expert_id = gate_up_eid

                # down bundle — use the standard preload() which is
                # already slot-aware.
                rt_down.preload(down_eid, gptq["down"], kernel_mode)
                preloaded += 1
            except Exception:
                self._record_fallback("speculative_preload_gptq_failed")
                # Keep going — partial warm-up is still useful.

        if preloaded > 0:
            self._speculative_preload_gptq_count = (
                getattr(self, "_speculative_preload_gptq_count", 0) + preloaded
            )

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

        # ADR-002 M-9: routing-locality diagnostic.  Compute Jaccard
        # similarity of this call's active CPU-side expert set vs the
        # previous call.  Must run unconditionally (even if we route
        # back to CPU) so decode-step locality is measured honestly.
        try:
            flat_ids_cpu = topk_ids.detach().to("cpu", dtype=torch.long).flatten()
            cpu_mask_cpu = (~self.gpu_experts_mask.bool()).to("cpu")
            # An expert is "active on CPU side" this call iff any token
            # routed to it AND it lives on CPU (not GPU-resident).
            active_cpu = {
                int(eid) for eid in flat_ids_cpu.unique().tolist()
                if 0 <= eid < self.num_experts and bool(cpu_mask_cpu[eid].item())
            }
            current = frozenset(active_cpu)
            prev = self._prev_active_cpu_experts_forward
            if prev is not None and (current or prev):
                union = len(current | prev)
                inter = len(current & prev)
                j = (inter / union) if union > 0 else 0.0
                if context.is_prefill:
                    self.locality_prefill_jaccard_sum += j
                    self.locality_prefill_jaccard_count += 1
                else:
                    self.locality_decode_jaccard_sum += j
                    self.locality_decode_jaccard_count += 1
                    # bucket into 11 bins, last bin = exactly 1.0
                    if j >= 1.0:
                        self.locality_decode_jaccard_histogram[10] += 1
                    else:
                        self.locality_decode_jaccard_histogram[int(j * 10)] += 1
            self._prev_active_cpu_experts_forward = current
        except Exception:
            # locality accounting must never break a forward pass.
            pass

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

        # ADR-002 M-25 Stage A: diagnostic counters moved out of
        # submit_forward (which used GPU .item() and caused a CUDA
        # sync every layer every step).  Non-async / fallback path
        # computes them here on CPU-materialised tensors.
        routed_cpu = cpu_mask[topk_ids_cpu]
        self.offloaded_pairs += int(routed_cpu.sum().item())
        self.offloaded_tokens += int(routed_cpu.any(dim=1).sum().item())

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

        # ADR-002 M-14: batch the common GPTQ quantized path before
        # falling back to the legacy per-expert loop.  The batched path
        # preserves Python routing/gather/index_add semantics but reduces
        # ctypes crossings by grouping gate+up runs and down runs.
        if self.is_gptq and self.quantized_runtime is not None and activated_cpu_experts:
            all_quantized = all(cpu_slot in self._gptq_experts for _expert_idx, cpu_slot, _ti, _m in activated_cpu_experts)
            if all_quantized:
                try:
                    if self._run_quantized_experts_batched_on_dpu(
                        activated_cpu_experts,
                        flat_cpu,
                        topk_weights_cpu,
                        output,
                    ):
                        self._fallback_output = output.to(device=device, dtype=hidden_states.dtype)
                        return True
                except Exception:
                    self._record_fallback("expert_quantized_dpu_batched_run_failed")

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
        # ADR-002 M-25 Stage A: eliminate GPU-side sync points in the
        # submit_forward hot path.
        #
        # The two removed ``.item()`` calls here used to run on the GPU
        # topk_ids tensor on every decode step of every layer (48*32 =
        # 1536 CUDA syncs per run), each one forcing the GPU expert
        # loop's first matmul to wait for router tensors to cross D2H.
        # M-24 profiling showed ~0.1 s of this 26-27 s decode was
        # ``.item()`` dispatch overhead and the rest was serialisation
        # of the GPU expert loop behind blocking host reads.
        #
        # offloaded_pairs / offloaded_tokens are diagnostic counters
        # only (they feed pim_compute_participation_ratio and ADR
        # science-integrity guards); they have zero effect on the
        # numerical result.  We therefore defer the accumulation to
        # ``_submit_forward_c_async`` (the hot path), which already
        # materialises topk_ids on CPU and can compute the same
        # counters without paying an extra sync.  Non-async fallback
        # paths (real/shadow sync, M-10 Python thread) still pay the
        # sync via ``_submit_forward_real``; see the counter update
        # co-located with its cpu_mask slice around line 1355.
        context = get_context()

        # ADR-002 M-24 Stage A: C-level async submit.
        #
        # When enabled and the layer qualifies (decode, GPTQ, real mode,
        # has CPU-side experts), push the whole gate_up+silu*up+down op
        # to a C pthread worker.  Python returns immediately; the GIL
        # stays released throughout DPU work so HybridMoE.forward's GPU
        # expert loop runs truly concurrently.  sync_forward later joins
        # the worker via _sync_forward_c_async.
        #
        # Falls back transparently to the legacy path on any failure.
        c_async_eligible = (
            self.enable_c_async_submit
            and self.pim_execution_mode == "real"
            and self.has_cpu_experts
            and self.is_gptq
            and not context.is_prefill
            and self.quantized_runtime is not None
        )
        if c_async_eligible:
            try:
                ok = self._submit_forward_c_async(
                    hidden_states, topk_ids, topk_weights
                )
                if ok:
                    return
                self.c_async_fallback_count += 1
            except Exception:
                self._record_fallback("c_async_submit_failed")
                self.c_async_fallback_count += 1
                # Drop any partial handle and fall through.
                self._c_async_handle = None
                self._c_async_meta = None

        # ADR-002 M-10: async PIM submit.
        #
        # When enabled and in decode (not prefill), spawn a background
        # thread that runs _submit_forward_real while HybridMoE.forward
        # proceeds with GPU attention / GPU-resident experts.
        # sync_forward joins the thread.  Prefill intentionally stays
        # synchronous because:
        #   * prefill usually routes through CPU (cost-model decision),
        #     so there's no DPU work to hide anyway;
        #   * prefill is called in a setup phase where GPU side has
        #     very little overlap budget.
        async_eligible = (
            self.enable_async_pim_submit
            and self.pim_execution_mode == "real"
            and self.has_cpu_experts
            and not context.is_prefill
        )

        if async_eligible:
            # Clear any prior thread state (safety — sync_forward should
            # have joined by now).  If a previous thread is still alive
            # we must join synchronously here to avoid leaking work.
            if self._async_thread is not None and self._async_thread.is_alive():
                self._async_thread.join()
            self._async_exc = None
            self._async_submit_wall_start = None

            # Snapshot the torch tensors so the background thread gets
            # its own view (the caller's hidden_states / topk_ids may
            # be freed or mutated after submit returns).
            hs_snap = hidden_states
            tk_ids_snap = topk_ids
            tk_w_snap = topk_weights

            def _worker():
                try:
                    ok = self._submit_forward_real(hs_snap, tk_ids_snap, tk_w_snap)
                    if not ok:
                        # cost-model voted CPU / no-op; populate zeros so
                        # sync_forward has a shape-matching tensor.
                        if (
                            self._fallback_output is None
                            or not isinstance(self._fallback_output, torch.Tensor)
                            or self._fallback_output.shape != hs_snap.shape
                        ):
                            self._fallback_output = torch.zeros_like(hs_snap)
                except BaseException as exc:  # noqa: BLE001
                    self._async_exc = exc

            import time as _time
            self._async_submit_wall_start = _time.perf_counter()
            t = threading.Thread(target=_worker, name=f"pim_async_L{self.layer_idx}", daemon=True)
            t.start()
            self._async_thread = t
            self.async_submit_count += 1
            return

        if self.pim_execution_mode == "real" and self.has_cpu_experts:
            if self._submit_forward_real(hidden_states, topk_ids, topk_weights):
                # If prefill fell through to CPU, speculatively preload for decode
                return

        # Prefill path: after CPU fallback, speculatively preload hottest expert
        if context.is_prefill and self.pim_execution_mode == "real":
            if self.expert_runtime is not None:
                self._speculative_preload(topk_ids)
            # ADR-002 M-7: GPTQ-aware warm-up of the layer-group slot
            # cache.  Without this, decode step 1 starts cold on every
            # layer and pays the full miss cost.
            if self.is_gptq and self.quantized_runtime is not None:
                self._speculative_preload_gptq(topk_ids)

        super().submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream: int | None) -> torch.Tensor:
        """ADR-002 M-10: join the async PIM submit thread (if any) and
        then delegate to the CPU/GPTQ sync path which reads
        ``self._fallback_output``.

        ADR-002 M-24 Stage A: if a C-level async handle is pending, join
        its pthread worker and return the assembled output directly.
        This keeps the GPU expert loop running in parallel with PIM DPU
        work on real hardware.

        ADR-002 M-26: if a background Python submit thread is still
        running (the thread that ran ``_do_c_async_submit_work`` in
        parallel with the main-thread GPU expert loop), join it first
        so ``self._c_async_handle`` is guaranteed populated before we
        dispatch to ``_sync_forward_c_async``.

        If neither async path was used this call, the straight delegate
        pays only one branch check.
        """
        # ADR-002 M-26: join the Python submit thread before touching
        # the C async handle.  This is the only place the main thread
        # waits for the preload + submit work to finish; if it is
        # short (expected steady-state ~300-500us) it overlapped with
        # the GPU expert loop.
        t26 = self._m26_submit_thread
        if t26 is not None:
            import time as _time
            wait_start = _time.perf_counter()
            t26.join()
            wait_s = _time.perf_counter() - wait_start
            self.m26_threaded_submit_wait_sum += wait_s
            self.m26_threaded_submit_wait_count += 1
            self._m26_submit_thread = None
            exc = self.m26_threaded_submit_exc
            self.m26_threaded_submit_exc = None
            if exc is not None:
                raise exc

        # Prefer C async path when a handle is pending (Stage A).
        if self._c_async_handle is not None:
            out = self._sync_forward_c_async()
            if out is not None:
                return out
            # _sync_forward_c_async returned None only if the handle was
            # cleared concurrently (not expected); fall through.

        t = self._async_thread
        if t is not None:
            import time as _time
            wait_start = _time.perf_counter()
            t.join()
            wait_s = _time.perf_counter() - wait_start
            self.async_sync_wait_seconds_sum += wait_s
            self.async_sync_wait_seconds_count += 1
            self._async_thread = None
            exc = self._async_exc
            self._async_exc = None
            if exc is not None:
                raise exc
        return super().sync_forward(hidden_states, cuda_stream)

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

        # Clean up quantized runtimes.  In M-5 the gate_up bundle and
        # down projection live on two independent runtimes; clear both.
        for rt in {
            id(r): r
            for r in (self.quantized_runtime, self.quantized_runtime_down)
            if r is not None
        }.values():
            try:
                for xor_mask in (
                    0x1111111111111111,
                    0x2222222222222222,
                    0x3333333333333333,
                    0x1212121212121212,  # M-4.1 fused gate+up bundle
                ):
                    proj_eid = eid ^ xor_mask
                    rt.evict_cached_weights(proj_eid)
            except Exception:
                pass

    def diagnostics(self) -> dict[str, Any]:
        profile_means: dict[str, Optional[float]] = {}
        for field, value in self.quantized_profile_seconds_sum_local.items():
            denom = (
                self.quantized_profile_load_count_local
                if field in PIMQuantizedRuntime.PROFILE_LOAD_FIELDS
                else self.quantized_profile_run_count_local
            )
            profile_means[field] = (value / denom) if denom > 0 else None

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
                # ADR-002 M-5: dual-runtime diagnostics.
                #
                # ``quantized_runtime_down_distinct`` is True iff M-5's
                # split allocation succeeded (two separate DPU rank
                # pools).  Backend-local preload counters are the
                # ground truth for this backend's behaviour; the
                # runtime-global counters above are shared across 48
                # layers and therefore only useful in aggregate.
                "quantized_runtime_down_distinct": (
                    self.quantized_runtime_down is not None
                    and self.quantized_runtime_down is not self.quantized_runtime
                ),
                "quantized_runtime_down_preload_hits": (
                    0 if self.quantized_runtime_down is None
                    else self.quantized_runtime_down.preload_hits
                ),
                "quantized_runtime_down_preload_misses": (
                    0 if self.quantized_runtime_down is None
                    else self.quantized_runtime_down.preload_misses
                ),
                "quantized_preload_hits_local": self.quantized_preload_hits_local,
                "quantized_preload_misses_local": self.quantized_preload_misses_local,
                # ADR-002 M-13: native PIM profile aggregation.  These
                # fields are local to this MoE layer, unlike runtime-global
                # counters on shared PIMQuantizedRuntime instances.
                "quantized_profile_load_count_local": self.quantized_profile_load_count_local,
                "quantized_profile_run_count_local": self.quantized_profile_run_count_local,
                "quantized_profile_seconds_sum_local": dict(self.quantized_profile_seconds_sum_local),
                "quantized_profile_seconds_mean_local": profile_means,
                "quantized_batched_expert_groups_local": self.quantized_batched_expert_groups_local,
                "quantized_batched_experts_local": self.quantized_batched_experts_local,
                # ADR-002 M-17.3: diagnostic counter for the
                # down-preload-before-gate-up-infer overlap.  Equals
                # batched_expert_groups when the dual-runtime split is
                # active everywhere; smaller when fallback paths kicked
                # in for some calls.
                "quantized_down_preload_overlap_local": self.quantized_down_preload_overlap_local,
                # ADR-002 M-7: layer-group scoping + GPTQ speculative preload.
                "pim_layer_group_size": self.pim_layer_group_size,
                "pim_layer_group_id": self.layer_idx // max(1, self.pim_layer_group_size),
                "enable_speculative_preload_gptq": self.enable_speculative_preload_gptq,
                "speculative_preload_gptq_count": getattr(
                    self, "_speculative_preload_gptq_count", 0
                ),
                # ADR-002 M-9: routing-locality diagnostic (Jaccard of
                # active CPU-side expert set between consecutive forward
                # calls).  Aggregated here so dev_gate / reports can do
                # cross-layer sums and means.
                "locality_decode_jaccard_count": self.locality_decode_jaccard_count,
                "locality_decode_jaccard_sum": self.locality_decode_jaccard_sum,
                "locality_decode_jaccard_mean": (
                    (self.locality_decode_jaccard_sum / self.locality_decode_jaccard_count)
                    if self.locality_decode_jaccard_count > 0
                    else None
                ),
                "locality_decode_jaccard_histogram": list(self.locality_decode_jaccard_histogram),
                "locality_prefill_jaccard_count": self.locality_prefill_jaccard_count,
                "locality_prefill_jaccard_mean": (
                    (self.locality_prefill_jaccard_sum / self.locality_prefill_jaccard_count)
                    if self.locality_prefill_jaccard_count > 0
                    else None
                ),
                # ADR-002 M-10: async PIM submit telemetry.
                "enable_async_pim_submit": self.enable_async_pim_submit,
                "async_submit_count": self.async_submit_count,
                "async_sync_wait_seconds_sum": self.async_sync_wait_seconds_sum,
                "async_sync_wait_seconds_count": self.async_sync_wait_seconds_count,
                "async_sync_wait_seconds_mean": (
                    (self.async_sync_wait_seconds_sum / self.async_sync_wait_seconds_count)
                    if self.async_sync_wait_seconds_count > 0
                    else None
                ),
                # ADR-002 M-24 Stage B: C-level fused kernel diagnostics.
                "enable_c_fused_kernel": self.enable_c_fused_kernel,
                "c_fused_calls": self.c_fused_calls,
                "c_fused_experts_processed": self.c_fused_experts_processed,
                "c_fused_fallback_count": self.c_fused_fallback_count,
                # ADR-002 M-24 Stage A: reserved flag (wired but inactive
                # until the C async submit path lands).
                "enable_c_async_submit": self.enable_c_async_submit,
                "c_async_submit_count": self.c_async_submit_count,
                "c_async_fallback_count": self.c_async_fallback_count,
                "c_async_sync_wait_seconds_sum": self.c_async_sync_wait_seconds_sum,
                "c_async_sync_wait_seconds_count": self.c_async_sync_wait_seconds_count,
                "c_async_sync_wait_seconds_mean": (
                    (self.c_async_sync_wait_seconds_sum / self.c_async_sync_wait_seconds_count)
                    if self.c_async_sync_wait_seconds_count > 0
                    else None
                ),
                # ADR-002 M-25 Stage A/B: pinned-buffer D2H/H2D to
                # eliminate GPU-side sync points in the hot path.
                "enable_m25_pinned_d2h": self.enable_m25_pinned_d2h,
                "m25_pinned_submit_cache_shapes": list(self._m25_pinned_cache.keys()),
                "m25_pinned_output_cache_shapes": list(self._m25_output_pinned_cache.keys()),
                # ADR-002 M-26: background submit thread diagnostics.
                "enable_m26_threaded_submit": self.enable_m26_threaded_submit,
                "m26_threaded_submit_count": self.m26_threaded_submit_count,
                "m26_threaded_submit_wait_sum": self.m26_threaded_submit_wait_sum,
                "m26_threaded_submit_wait_count": self.m26_threaded_submit_wait_count,
                "m26_threaded_submit_wait_mean": (
                    (self.m26_threaded_submit_wait_sum / self.m26_threaded_submit_wait_count)
                    if self.m26_threaded_submit_wait_count > 0
                    else None
                ),
                # PIM real-compute participation ratio.  Offloaded experts
                # are those routed to a non-GPU-resident slot; PIM-computed
                # experts are those handled by one of the DPU paths
                # (legacy batched, c_fused, or the per-expert quantized
                # loop).  This is the science-integrity guard for M-24:
                # benchmarks that accidentally bypass PIM will surface as
                # this ratio dropping below 0.7 and fail dev_gate.
                "pim_compute_participation_ratio": (
                    (self.real_dpu_expert_calls / self.offloaded_tokens)
                    if self.offloaded_tokens > 0
                    else None
                ),
            }
        )
        return diagnostics
