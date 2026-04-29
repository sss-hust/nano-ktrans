from __future__ import annotations

import atexit
import ctypes
import fcntl
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch

from .weight_loader import GPTQLinearWeight


class PIMQuantizedRuntime:
    BLOCK_FLOATS = 64
    VALUES_PER_WORD = 8
    MAX_QWEIGHT_WORDS = 2_097_152
    MAX_SCALE_FLOATS = 65_536
    MAX_INPUT_FLOATS = 65_536
    MAX_OUTPUT_FLOATS = 65_536
    ERROR_BUFFER_SIZE = 2048

    # ADR-002 M-6.1: multi-slot MRAM residency.  Must match NUM_SLOTS
    # in dpu_quantized_kernel.c and host_quantized_bridge.c.  Keeping
    # N > top_k (Qwen3 uses top_k=8) lets a decode step that revisits
    # an expert in the next step find its bundle still resident.
    NUM_SLOTS = 8
    PROFILE_LOAD_FIELDS = (
        "load_qweight_transfer_seconds",
        "load_scale_transfer_seconds",
        "load_total_seconds",
    )
    PROFILE_RUN_FIELDS = (
        "input_transfer_seconds",
        "launch_seconds",
        "output_transfer_seconds",
        "runtime_total_seconds",
    )

    _shared: dict[tuple[str, int], "PIMQuantizedRuntime"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, *, profile: str = "", rank_count: int = 1, instance_key: str = "") -> None:
        # ``profile`` is passed through to UPMEM's ``dpu_alloc_ranks``
        # and must be empty or a valid UPMEM profile string (e.g.
        # ``"backend=hw"``).  Do NOT stuff logical routing tags in
        # here — UPMEM rejects unknown keys with "invalid profile".
        #
        # ADR-002 M-8: ``instance_key`` is the Python-side cache key
        # for ``get_shared``.  Callers that want multiple *physical*
        # DPU rank pools (M-5 dual runtime, M-7 per-layer-group) vary
        # ``instance_key`` while keeping ``profile`` empty; each
        # distinct key now really does allocate a new DPU rank pool
        # because pim_quantized_init() in the handle-based bridge
        # always calls dpu_alloc_ranks.
        self.profile = profile
        self.instance_key = instance_key or profile
        self.rank_count = rank_count
        native_dir = Path(__file__).resolve().parent / "pim_native"
        build_dir = native_dir / "build"
        self._build_if_needed(native_dir, build_dir)

        self.kernel_path = build_dir / "pim_quantized_kernel"
        self.lib_path = build_dir / "libpim_quantized_bridge.so"
        if not self.kernel_path.exists() or not self.lib_path.exists():
            raise RuntimeError("PIM quantized bridge build did not produce the expected artifacts.")

        self._lib = ctypes.CDLL(str(self.lib_path))
        # ADR-002 M-8: every function now takes a handle (ctypes.c_void_p)
        # as the first argument so the .so no longer relies on shared
        # `static` globals (see ADR-002 §15 for the bug root-causing
        # M-5/M-6/M-7 null perf results).
        self._lib.pim_quantized_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_init.restype = ctypes.c_void_p   # handle
        self._lib.pim_quantized_load_weights.argtypes = [
            ctypes.c_void_p,   # handle
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,   # slot_id (ADR-002 M-6.1)
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_load_weights.restype = ctypes.c_int
        self._lib.pim_quantized_run.argtypes = [
            ctypes.c_void_p,   # handle
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,   # slot_id (ADR-002 M-6.1)
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_run.restype = ctypes.c_int
        self._lib.pim_quantized_last_cycles.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_cycles.restype = ctypes.c_uint64
        self._lib.pim_quantized_last_load_qweight_transfer_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_load_qweight_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_load_scale_transfer_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_load_scale_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_load_total_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_load_total_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_input_transfer_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_input_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_launch_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_launch_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_output_transfer_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_output_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_total_seconds.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_last_total_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_num_dpus.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_num_dpus.restype = ctypes.c_uint32
        self._lib.pim_quantized_shutdown.argtypes = [ctypes.c_void_p]
        self._lib.pim_quantized_shutdown.restype = None

        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        handle = self._lib.pim_quantized_init(
            os.fsencode(str(self.kernel_path)),
            profile.encode() if profile else None,
            ctypes.c_uint32(rank_count),
            error_buffer,
            len(error_buffer),
        )
        if not handle:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        self._handle = ctypes.c_void_p(handle)
        self._loaded_signature: tuple[int, int, int, int, int] | None = None

        # ── Weight residency tracking (NEW) ────────────────────────────
        self._resident_expert_id: int = 0

        # Cached pre-padded quantized weights: expert_id → (qweight_i32, scales_f32, padded_input_dim, padded_output_dim, group_size, kernel_mode, original_output_dim)
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]] = {}

        # ADR-002 M-6.1: per-slot residency + LRU bookkeeping.
        #
        # ``_slot_expert_id[slot]`` == 0 means the slot is empty; any other
        # value is the expert_id whose weights are currently in that
        # slot.  ``_slot_lru`` is a rolling counter so eviction picks
        # the slot with the oldest touch.  ``_expert_to_slot`` is the
        # inverse map used for O(1) hit detection on ``preload``.
        self._slot_expert_id: list[int] = [0] * self.NUM_SLOTS
        self._slot_lru_ticker: list[int] = [0] * self.NUM_SLOTS
        self._expert_to_slot: dict[int, int] = {}
        self._lru_counter: int = 0
        # ``_last_touched_slot`` remembers which slot the last preload
        # landed in; infer() uses it when no explicit slot_id is passed.
        self._last_touched_slot: int = 0

        # Stats
        self.preload_hits: int = 0
        self.preload_misses: int = 0

        # ADR-002 M-13: cumulative native profile counters.  The C bridge
        # exposes only "last call" timings; aggregate them here so callers
        # can take before/after deltas and attribute per-layer PIM time.
        self._profile_totals: dict[str, float] = {
            field: 0.0 for field in (*self.PROFILE_LOAD_FIELDS, *self.PROFILE_RUN_FIELDS)
        }
        self._profile_load_count: int = 0
        self._profile_run_count: int = 0

    @classmethod
    def get_shared(
        cls,
        *,
        profile: str = "",
        rank_count: int = 1,
        instance_key: str = "",
    ) -> "PIMQuantizedRuntime":
        """ADR-002 M-8: ``instance_key`` is the cache discriminator
        (defaults to ``profile`` for backward compatibility with
        callers that did not know about the split yet).  ``profile``
        is what UPMEM actually sees in dpu_alloc_ranks — keep it empty
        unless you actually need UPMEM-level profile tuning.
        """
        effective_key = instance_key or profile
        key = (effective_key, rank_count)
        with cls._shared_lock:
            runtime = cls._shared.get(key)
            if runtime is None:
                runtime = cls(
                    profile=profile,
                    rank_count=rank_count,
                    instance_key=effective_key,
                )
                cls._shared[key] = runtime
                atexit.register(runtime.shutdown)
            return runtime

    @classmethod
    def _build_if_needed(cls, native_dir: Path, build_dir: Path) -> None:
        kernel_path = build_dir / "pim_quantized_kernel"
        lib_path = build_dir / "libpim_quantized_bridge.so"
        lock_path = build_dir / ".build.lock"
        sources = [
            native_dir / "build.sh",
            native_dir / "dpu_quantized_kernel.c",
            native_dir / "host_quantized_bridge.c",
        ]
        build_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if kernel_path.exists() and lib_path.exists():
                newest_source = max(path.stat().st_mtime for path in sources)
                oldest_artifact = min(kernel_path.stat().st_mtime, lib_path.stat().st_mtime)
                if oldest_artifact >= newest_source:
                    return

            subprocess.run(
                ["bash", str(native_dir / "build.sh")],
                check=True,
                cwd=str(native_dir),
                env=os.environ.copy(),
            )

    @classmethod
    def supports_shape(cls, batch_size: int, input_dim: int, output_dim: int, group_size: int) -> bool:
        if batch_size <= 0 or input_dim <= 0 or output_dim <= 0 or group_size <= 0:
            return False
        padded_input_dim = ((input_dim + cls.BLOCK_FLOATS - 1) // cls.BLOCK_FLOATS) * cls.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)
        qweight_words = (padded_input_dim // cls.VALUES_PER_WORD) * padded_output_dim
        if (padded_input_dim % group_size) != 0:
            return False
        scale_floats = (padded_input_dim // group_size) * padded_output_dim
        return (
            qweight_words <= cls.MAX_QWEIGHT_WORDS
            and
            scale_floats <= cls.MAX_SCALE_FLOATS
            and batch_size * padded_input_dim <= cls.MAX_INPUT_FLOATS
            and batch_size * padded_output_dim <= cls.MAX_OUTPUT_FLOATS
        )

    # ── Weight preparation (shared padding logic) ──────────────────────

    def _prepare_quantized_weights(
        self,
        quantized: GPTQLinearWeight,
        kernel_mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        Pad quantized weight tensors for DPU consumption.
        Returns (qweight_i32, scales_f32, padded_input_dim, padded_output_dim, original_output_dim).
        """
        input_dim = quantized.input_dim
        output_dim = quantized.output_dim

        qweight_i32 = quantized.qweight.detach().to(device="cpu", dtype=torch.int32, copy=True).contiguous()
        scales_f32 = quantized.scales.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()

        padded_input_dim = ((input_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)

        if padded_input_dim != input_dim:
            words_per_row = padded_input_dim // quantized.values_per_word
            padded_qweight = torch.zeros(output_dim, words_per_row, dtype=torch.int32)
            padded_qweight[:, :qweight_i32.shape[1]] = qweight_i32
            qweight_i32 = padded_qweight

        if padded_output_dim != output_dim:
            padded_qweight_out = torch.zeros(padded_output_dim, qweight_i32.shape[1], dtype=torch.int32)
            padded_qweight_out[:output_dim, :] = qweight_i32
            qweight_i32 = padded_qweight_out

            padded_scales = torch.zeros(padded_output_dim, scales_f32.shape[1], dtype=torch.float32)
            padded_scales[:output_dim, :] = scales_f32
            scales_f32 = padded_scales

        return qweight_i32, scales_f32, padded_input_dim, padded_output_dim, output_dim

    # ADR-002 M-4.1: host-side concat of two shape-compatible GPTQ
    # projections (same input_dim, same group_size) into one fat
    # projection.  Used by PIMMoEBackend to merge gate + up into a
    # single DPU call per expert — the DPU kernel is agnostic to
    # output_dim, so this is purely a host-side weight stacking.
    #
    # Returns the same tuple as _prepare_quantized_weights so the
    # existing preload path can consume it unchanged.
    def _prepare_concat_quantized_weights(
        self,
        lhs: GPTQLinearWeight,
        rhs: GPTQLinearWeight,
        kernel_mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        if lhs.input_dim != rhs.input_dim:
            raise ValueError(
                f"concat weights must share input_dim; "
                f"got lhs={lhs.input_dim}, rhs={rhs.input_dim}"
            )
        if lhs.group_size != rhs.group_size:
            raise ValueError(
                f"concat weights must share group_size; "
                f"got lhs={lhs.group_size}, rhs={rhs.group_size}"
            )
        if lhs.bits != rhs.bits:
            raise ValueError(
                f"concat weights must share bit-width; "
                f"got lhs={lhs.bits}, rhs={rhs.bits}"
            )

        # Prepare each side individually (padding applied) then stack
        # along the output-dim axis.  This avoids duplicating the
        # padding logic while keeping the memory-layout invariants
        # the DPU kernel expects.
        lhs_qw, lhs_sc, padded_in, lhs_padded_out, lhs_orig_out = \
            self._prepare_quantized_weights(lhs, kernel_mode)
        rhs_qw, rhs_sc, _padded_in_r, rhs_padded_out, rhs_orig_out = \
            self._prepare_quantized_weights(rhs, kernel_mode)

        # Both sides share padded_input_dim by construction (same
        # input_dim).  Stack rows.
        concat_qw = torch.cat([lhs_qw, rhs_qw], dim=0).contiguous()
        concat_sc = torch.cat([lhs_sc, rhs_sc], dim=0).contiguous()
        # Re-pad output_dim to even (DPU kernel writes row pairs).
        concat_rows = concat_qw.shape[0]
        if concat_rows % 2 == 1:
            concat_qw = torch.cat(
                [concat_qw, torch.zeros(1, concat_qw.shape[1], dtype=torch.int32)], dim=0
            ).contiguous()
            concat_sc = torch.cat(
                [concat_sc, torch.zeros(1, concat_sc.shape[1], dtype=torch.float32)], dim=0
            ).contiguous()
            concat_rows += 1
        # original_lhs_out and original_rhs_out are returned so the
        # caller can split the DPU output back into two tensors.
        return (
            concat_qw,
            concat_sc,
            padded_in,
            concat_rows,
            lhs_orig_out,
            rhs_orig_out,
        )

    # ── NEW: preload — load quantized weights to DPU MRAM ──────────────

    def _touch_slot(self, slot: int) -> None:
        """Bump LRU tick on a slot (called on hit and on fresh load)."""
        self._lru_counter += 1
        self._slot_lru_ticker[slot] = self._lru_counter

    def _allocate_slot(self, expert_id: int) -> tuple[int, bool]:
        """
        Return ``(slot_id, was_resident)`` for this expert.  If the
        expert already lives in some slot, returns that slot with
        ``was_resident=True`` and the LRU tick is bumped.  Otherwise
        picks the LRU-oldest slot, evicts whatever was there, and
        returns ``(slot, False)`` — the caller is responsible for
        uploading weights into the slot via ``pim_quantized_load_weights``.
        """
        slot = self._expert_to_slot.get(expert_id)
        if slot is not None:
            self._touch_slot(slot)
            return slot, True

        # Pick an empty slot if any (prefer slot 1 .. N-1 over slot 0
        # only to keep slot 0 empty during init; any slot works once
        # we're past bootstrap).  Empty == _slot_expert_id[slot] == 0.
        chosen: int = -1
        for s in range(self.NUM_SLOTS):
            if self._slot_expert_id[s] == 0:
                chosen = s
                break
        if chosen < 0:
            # All slots full — evict least-recently-touched.
            chosen = min(range(self.NUM_SLOTS), key=lambda s: self._slot_lru_ticker[s])
            old_eid = self._slot_expert_id[chosen]
            if old_eid in self._expert_to_slot:
                del self._expert_to_slot[old_eid]

        self._slot_expert_id[chosen] = expert_id
        self._expert_to_slot[expert_id] = chosen
        self._touch_slot(chosen)
        return chosen, False

    def _evict_from_slots(self, expert_id: int) -> None:
        """Drop ``expert_id`` from the slot table if present."""
        slot = self._expert_to_slot.pop(expert_id, None)
        if slot is not None:
            self._slot_expert_id[slot] = 0
            self._slot_lru_ticker[slot] = 0

    def preload(
        self,
        expert_id: int,
        quantized: GPTQLinearWeight,
        kernel_mode: int = 4,
    ) -> bool:
        """
        Load quantized expert weights to DPU MRAM if not already resident.

        Returns True if weights were actually transferred (cache miss),
        False if they were already resident in some slot (cache hit).

        ADR-002 M-6.1: this path now maintains an 8-slot LRU cache
        backed by the DPU's multi-slot MRAM layout.  A hit takes zero
        host->DPU DMA; a miss transfers only into the slot chosen by
        ``_allocate_slot``.
        """
        slot, was_resident = self._allocate_slot(expert_id)
        self._last_touched_slot = slot

        if was_resident:
            # Also keep legacy single-slot state in sync so callers
            # inspecting ``resident_expert_id`` see a sensible value.
            self._resident_expert_id = expert_id
            self.preload_hits += 1
            return False

        # Miss path: make sure the host-side padded cache exists then
        # upload to the chosen slot.
        if expert_id not in self._weight_cache:
            qweight_i32, scales_f32, padded_input_dim, padded_output_dim, orig_output_dim = \
                self._prepare_quantized_weights(quantized, kernel_mode)
            self._weight_cache[expert_id] = (
                qweight_i32, scales_f32, padded_input_dim, padded_output_dim,
                quantized.group_size, kernel_mode, orig_output_dim,
            )

        qweight_i32, scales_f32, padded_input_dim, padded_output_dim, group_size, km, _orig = \
            self._weight_cache[expert_id]

        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_load_weights(
            self._handle,
            ctypes.c_uint32(padded_input_dim),
            ctypes.c_uint32(padded_output_dim),
            ctypes.c_uint32(group_size),
            ctypes.c_uint32(km),
            ctypes.c_void_p(qweight_i32.data_ptr()),
            ctypes.c_void_p(scales_f32.data_ptr()),
            ctypes.c_uint32(slot),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))

        # Legacy single-slot fields — kept for backward compatibility
        # with anything that still reads them.  Semantics change
        # slightly: ``_resident_expert_id`` is now the most-recently
        # preloaded expert rather than the single occupant.
        self._record_load_profile()
        self._loaded_signature = (padded_input_dim, padded_output_dim, group_size, km, expert_id)
        self._resident_expert_id = expert_id
        self.preload_misses += 1
        return True

    # ── NEW: infer — run DPU quantized kernel with resident weights ────

    def infer(self, inputs: torch.Tensor, *, slot_id: Optional[int] = None) -> torch.Tensor:
        """
        Run the DPU quantized kernel using weights already resident in MRAM.
        Only transfers input activations to DPU.

        ADR-002 M-6.1: ``slot_id`` selects which MRAM slot to compute
        from.  If omitted, we use the slot touched by the most recent
        ``preload()`` — which is the right default for the common
        ``preload(eid); infer()`` pair and keeps the pre-M-6 API shape.
        """
        if self._resident_expert_id == 0:
            raise RuntimeError("No quantized weights are resident. Call preload() first.")

        if inputs.ndim != 2:
            raise ValueError("PIMQuantizedRuntime.infer() expects 2D input tensor.")

        effective_slot = int(slot_id) if slot_id is not None else self._last_touched_slot
        if not (0 <= effective_slot < self.NUM_SLOTS):
            raise ValueError(
                f"slot_id must be in [0, {self.NUM_SLOTS}); got {effective_slot}"
            )

        batch_size, input_dim = inputs.shape
        inputs_f32 = inputs.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()

        # Resolve which expert's padded-weight cache entry describes
        # this slot's shape (caller may have preloaded a different
        # expert after this one).
        slot_expert = self._slot_expert_id[effective_slot]
        cache_key = slot_expert if slot_expert in self._weight_cache else self._resident_expert_id
        _, _, padded_input_dim, padded_output_dim, _, _, orig_output_dim = \
            self._weight_cache[cache_key]

        if padded_input_dim != input_dim:
            padded_inputs = torch.zeros(batch_size, padded_input_dim, dtype=torch.float32)
            padded_inputs[:, :input_dim] = inputs_f32
            inputs_f32 = padded_inputs

        outputs = torch.empty(batch_size, padded_output_dim, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_run(
            self._handle,
            ctypes.c_uint32(batch_size),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            ctypes.c_uint32(effective_slot),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        self._record_run_profile()

        return outputs[:, :orig_output_dim].contiguous()

    # ── NEW: evict — clear DPU weight residency ────────────────────────

    def evict(self) -> None:
        """Clear the current DPU quantized weight residency."""
        self._resident_expert_id = 0
        self._loaded_signature = None
        # M-6.1: also drop all slot residency so nothing stale lingers.
        self._slot_expert_id = [0] * self.NUM_SLOTS
        self._slot_lru_ticker = [0] * self.NUM_SLOTS
        self._expert_to_slot.clear()
        self._lru_counter = 0
        self._last_touched_slot = 0

    def evict_cached_weights(self, expert_id: int) -> None:
        """Remove cached pre-padded weights for a specific expert."""
        self._weight_cache.pop(expert_id, None)
        # M-6.1: drop the slot table entry too so a future preload of
        # a fresh expert_id is not confused by stale bookkeeping.
        self._evict_from_slots(expert_id)
        if self._resident_expert_id == expert_id:
            self._resident_expert_id = 0
            self._loaded_signature = None

    # ── NEW (ADR-002 M-4.1): fused preload+infer for two concatenated
    #    projections sharing the same input.  One DPU launch instead
    #    of two, one host->DPU weight transfer instead of two.
    def preload_and_infer_concat(
        self,
        expert_id: int,
        lhs: GPTQLinearWeight,
        rhs: GPTQLinearWeight,
        inputs: torch.Tensor,
        *,
        kernel_mode: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single DPU matvec on the row-concatenation of two
        quantized projections and split the result.

        Used by PIMMoEBackend to fuse gate + up into one DPU launch
        per expert.  The DPU kernel is agnostic to output_dim so this
        is purely a host-side weight stacking trick — it halves the
        per-expert DPU launch count and halves the host->DPU weight
        transfers without touching the DPU binary.

        Returns ``(lhs_output, rhs_output)`` in the original shapes.
        """
        if inputs.ndim != 2:
            raise ValueError("preload_and_infer_concat expects 2D input.")

        batch_size, input_dim = inputs.shape
        if input_dim != lhs.input_dim or input_dim != rhs.input_dim:
            raise ValueError(
                f"input_dim mismatch: inputs.shape={tuple(inputs.shape)}, "
                f"lhs.input_dim={lhs.input_dim}, rhs.input_dim={rhs.input_dim}"
            )

        # Cache lookup: each expert_id may hold a pre-prepared concat bundle.
        if expert_id not in self._weight_cache or len(self._weight_cache[expert_id]) != 7 \
                or self._weight_cache[expert_id][6] <= 0:
            concat_qw, concat_sc, padded_in, concat_rows, lhs_orig, rhs_orig = \
                self._prepare_concat_quantized_weights(lhs, rhs, kernel_mode)
            # Store with a 7th slot = rhs_orig_out (encoded as positive int).
            # The single-projection cache entry uses this slot to hold
            # ``original_output_dim`` (same semantic), so we overload the
            # tuple shape here for concat bundles: when slot[6] > 0 and
            # slot[5] == kernel_mode we know it's a concat bundle; downstream
            # readers care only about tensor+shape.
            self._weight_cache[expert_id] = (
                concat_qw, concat_sc, padded_in, concat_rows,
                lhs.group_size, kernel_mode, rhs_orig,
            )
            concat_lhs_orig = lhs_orig
            concat_rhs_orig = rhs_orig
        else:
            concat_qw, concat_sc, padded_in, concat_rows, group_size, km, concat_rhs_orig = \
                self._weight_cache[expert_id]
            concat_lhs_orig = lhs.output_dim

        # ADR-002 M-6.1: route through slot LRU — if this expert's
        # concat bundle is already resident in some slot, skip the
        # host->DPU transfer entirely.
        slot, was_resident = self._allocate_slot(expert_id)
        self._last_touched_slot = slot

        if not was_resident:
            error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
            rc = self._lib.pim_quantized_load_weights(
                self._handle,
                ctypes.c_uint32(padded_in),
                ctypes.c_uint32(concat_rows),
                ctypes.c_uint32(lhs.group_size),
                ctypes.c_uint32(kernel_mode),
                ctypes.c_void_p(concat_qw.data_ptr()),
                ctypes.c_void_p(concat_sc.data_ptr()),
                ctypes.c_uint32(slot),
                error_buffer,
                len(error_buffer),
            )
            if rc != 0:
                raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
            self._record_load_profile()
            self._resident_expert_id = expert_id
            self._loaded_signature = (padded_in, concat_rows, lhs.group_size, kernel_mode, expert_id)
            self.preload_misses += 1
        else:
            # Hit: keep ``_resident_expert_id`` pointing at this bundle
            # so infer()'s shape lookup finds it.
            self._resident_expert_id = expert_id
            self.preload_hits += 1

        # Pad inputs if needed.
        inputs_f32 = inputs.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        if padded_in != input_dim:
            padded_inputs = torch.zeros(batch_size, padded_in, dtype=torch.float32)
            padded_inputs[:, :input_dim] = inputs_f32
            inputs_f32 = padded_inputs

        outputs = torch.empty(batch_size, concat_rows, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_run(
            self._handle,
            ctypes.c_uint32(batch_size),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            ctypes.c_uint32(slot),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        self._record_run_profile()

        # Split: first concat_lhs_orig rows go to lhs, next concat_rhs_orig to rhs.
        lhs_out = outputs[:, :concat_lhs_orig].contiguous()
        rhs_out = outputs[:, concat_lhs_orig : concat_lhs_orig + concat_rhs_orig].contiguous()
        return lhs_out, rhs_out

    # ── Existing API (backward compatible) ──────────────────────────────

    def linear(self, inputs: torch.Tensor, quantized: GPTQLinearWeight, *, kernel_mode: int = 0) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"Expected 2D inputs, got shape={tuple(inputs.shape)}")
        batch_size, input_dim = inputs.shape
        if input_dim != quantized.input_dim:
            raise ValueError(
                f"Input dim mismatch: inputs.shape={tuple(inputs.shape)}, quantized.input_dim={quantized.input_dim}"
            )
        output_dim = quantized.output_dim
        if not self.supports_shape(batch_size, input_dim, output_dim, quantized.group_size):
            raise RuntimeError(
                f"PIM quantized shape unsupported: batch={batch_size}, input_dim={input_dim}, output_dim={output_dim}"
            )

        inputs_f32 = inputs.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        qweight_i32 = quantized.qweight.detach().to(device="cpu", dtype=torch.int32, copy=True).contiguous()
        scales_f32 = quantized.scales.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()

        padded_input_dim = ((input_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)
        if padded_input_dim != input_dim:
            padded_inputs = torch.zeros(batch_size, padded_input_dim, dtype=torch.float32)
            padded_inputs[:, :input_dim] = inputs_f32
            inputs_f32 = padded_inputs
            words_per_row = padded_input_dim // quantized.values_per_word
            padded_qweight = torch.zeros(output_dim, words_per_row, dtype=torch.int32)
            padded_qweight[:, : qweight_i32.shape[1]] = qweight_i32
            qweight_i32 = padded_qweight

        if padded_output_dim != output_dim:
            padded_qweight_out = torch.zeros(padded_output_dim, qweight_i32.shape[1], dtype=torch.int32)
            padded_qweight_out[:output_dim, :] = qweight_i32
            qweight_i32 = padded_qweight_out

            padded_scales = torch.zeros(padded_output_dim, scales_f32.shape[1], dtype=torch.float32)
            padded_scales[:output_dim, :] = scales_f32
            scales_f32 = padded_scales

        signature = (padded_input_dim, padded_output_dim, quantized.group_size, kernel_mode, id(quantized))
        if self._loaded_signature != signature:
            error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
            rc = self._lib.pim_quantized_load_weights(
                self._handle,
                ctypes.c_uint32(padded_input_dim),
                ctypes.c_uint32(padded_output_dim),
                ctypes.c_uint32(quantized.group_size),
                ctypes.c_uint32(kernel_mode),
                ctypes.c_void_p(qweight_i32.data_ptr()),
                ctypes.c_void_p(scales_f32.data_ptr()),
                ctypes.c_uint32(0),  # legacy .linear() always uses slot 0
                error_buffer,
                len(error_buffer),
            )
            if rc != 0:
                raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
            self._record_load_profile()
            self._loaded_signature = signature

        outputs = torch.empty(batch_size, padded_output_dim, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_run(
            self._handle,
            ctypes.c_uint32(batch_size),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            ctypes.c_uint32(0),  # legacy .linear() always uses slot 0
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        self._record_run_profile()
        return outputs[:, :output_dim].contiguous()

    # ── Query / diagnostics ─────────────────────────────────────────────

    @property
    def resident_expert_id(self) -> int:
        return self._resident_expert_id

    def last_cycles(self) -> int:
        return int(self._lib.pim_quantized_last_cycles(self._handle))

    def last_profile(self) -> dict[str, float]:
        return {
            "load_qweight_transfer_seconds": float(self._lib.pim_quantized_last_load_qweight_transfer_seconds(self._handle)),
            "load_scale_transfer_seconds": float(self._lib.pim_quantized_last_load_scale_transfer_seconds(self._handle)),
            "load_total_seconds": float(self._lib.pim_quantized_last_load_total_seconds(self._handle)),
            "input_transfer_seconds": float(self._lib.pim_quantized_last_input_transfer_seconds(self._handle)),
            "launch_seconds": float(self._lib.pim_quantized_last_launch_seconds(self._handle)),
            "output_transfer_seconds": float(self._lib.pim_quantized_last_output_transfer_seconds(self._handle)),
            "runtime_total_seconds": float(self._lib.pim_quantized_last_total_seconds(self._handle)),
        }

    def _record_load_profile(self) -> None:
        profile = self.last_profile()
        for field in self.PROFILE_LOAD_FIELDS:
            self._profile_totals[field] += float(profile.get(field, 0.0) or 0.0)
        self._profile_load_count += 1

    def _record_run_profile(self) -> None:
        profile = self.last_profile()
        for field in self.PROFILE_RUN_FIELDS:
            self._profile_totals[field] += float(profile.get(field, 0.0) or 0.0)
        self._profile_run_count += 1

    def profile_counters(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "load_count": self._profile_load_count,
            "run_count": self._profile_run_count,
        }
        for field, value in self._profile_totals.items():
            result[f"{field}_sum"] = float(value)
        return result

    def num_dpus(self) -> int:
        return int(self._lib.pim_quantized_num_dpus(self._handle))

    def shutdown(self) -> None:
        # ADR-002 M-8: handle becomes invalid after the underlying
        # pim_quantized_shutdown() frees its ctx; guard against double
        # free by zeroing the handle first.  Subsequent getters will
        # return 0 / 0.0 (per the NULL checks in host_quantized_bridge.c).
        handle = self._handle
        self._handle = ctypes.c_void_p(0)
        if handle:
            self._lib.pim_quantized_shutdown(handle)
        self._loaded_signature = None
        self._resident_expert_id = 0
        self._weight_cache.clear()
        # M-6.1: drop all slot-table state so a future (re-)init starts clean.
        self._slot_expert_id = [0] * self.NUM_SLOTS
        self._slot_lru_ticker = [0] * self.NUM_SLOTS
        self._expert_to_slot.clear()
        self._lru_counter = 0
        self._last_touched_slot = 0
