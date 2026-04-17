from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch


class PIMExpertRuntime:
    BLOCK_FLOATS = 64
    MAX_INTERMEDIATE_DIM = 2_048
    MAX_WEIGHT_FLOATS = 2_097_152
    MAX_INPUT_FLOATS = 65_536
    MAX_OUTPUT_FLOATS = 65_536
    ERROR_BUFFER_SIZE = 2048

    _shared: dict[tuple[str, int], "PIMExpertRuntime"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, *, profile: str = "", rank_count: int = 1) -> None:
        self.profile = profile
        self.rank_count = rank_count
        native_dir = Path(__file__).resolve().parent / "pim_native"
        build_dir = native_dir / "build"
        self._build_if_needed(native_dir, build_dir)

        self.kernel_path = build_dir / "pim_expert_kernel"
        self.lib_path = build_dir / "libpim_expert_bridge.so"
        if not self.kernel_path.exists() or not self.lib_path.exists():
            raise RuntimeError("PIM expert bridge build did not produce the expected artifacts.")

        self._lib = ctypes.CDLL(str(self.lib_path))

        # ── Existing API bindings ──────────────────────────────────────
        self._lib.pim_expert_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_expert_init.restype = ctypes.c_int
        self._lib.pim_expert_run.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_expert_run.restype = ctypes.c_int
        self._lib.pim_expert_last_cycles.argtypes = []
        self._lib.pim_expert_last_cycles.restype = ctypes.c_uint64
        self._lib.pim_expert_num_dpus.argtypes = []
        self._lib.pim_expert_num_dpus.restype = ctypes.c_uint32
        self._lib.pim_expert_last_active_dpus.argtypes = []
        self._lib.pim_expert_last_active_dpus.restype = ctypes.c_uint32
        self._lib.pim_expert_shutdown.argtypes = []
        self._lib.pim_expert_shutdown.restype = None

        # ── NEW: preload / infer / evict / resident_id bindings ────────
        self._lib.pim_expert_preload.argtypes = [
            ctypes.c_uint64,   # expert_id
            ctypes.c_uint32,   # input_dim
            ctypes.c_uint32,   # intermediate_dim
            ctypes.c_uint32,   # output_dim
            ctypes.c_void_p,   # gate_proj
            ctypes.c_void_p,   # up_proj
            ctypes.c_void_p,   # down_proj
            ctypes.c_char_p,   # error_buffer
            ctypes.c_size_t,   # error_buffer_len
        ]
        self._lib.pim_expert_preload.restype = ctypes.c_int

        self._lib.pim_expert_infer.argtypes = [
            ctypes.c_uint32,   # batch_size
            ctypes.c_void_p,   # inputs
            ctypes.c_void_p,   # outputs
            ctypes.c_char_p,   # error_buffer
            ctypes.c_size_t,   # error_buffer_len
        ]
        self._lib.pim_expert_infer.restype = ctypes.c_int

        self._lib.pim_expert_evict.argtypes = []
        self._lib.pim_expert_evict.restype = None

        self._lib.pim_expert_resident_id.argtypes = []
        self._lib.pim_expert_resident_id.restype = ctypes.c_uint64

        # ── DPU initialization ─────────────────────────────────────────
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_expert_init(
            os.fsencode(str(self.kernel_path)),
            profile.encode() if profile else None,
            ctypes.c_uint32(rank_count),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))

        # ── Python-side residency tracking (mirrors C-side) ────────────
        self._resident_expert_id: int = 0

        # Cached pre-padded f32 weight tensors: expert_id → (gate_f32, up_f32, down_f32, padded_dims, original_output_dim)
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], int]] = {}

        # Stats
        self.preload_hits: int = 0
        self.preload_misses: int = 0

    @classmethod
    def get_shared(cls, *, profile: str = "", rank_count: int = 1) -> "PIMExpertRuntime":
        key = (profile, rank_count)
        with cls._shared_lock:
            runtime = cls._shared.get(key)
            if runtime is None:
                runtime = cls(profile=profile, rank_count=rank_count)
                cls._shared[key] = runtime
                atexit.register(runtime.shutdown)
            return runtime

    @classmethod
    def _build_if_needed(cls, native_dir: Path, build_dir: Path) -> None:
        kernel_path = build_dir / "pim_expert_kernel"
        lib_path = build_dir / "libpim_expert_bridge.so"
        sources = [
            native_dir / "build.sh",
            native_dir / "dpu_expert_kernel.c",
            native_dir / "host_expert_bridge.c",
            native_dir / "silu_lut_4096.h",
        ]
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
    def supports_shape(cls, batch_size: int, input_dim: int, intermediate_dim: int, output_dim: int) -> bool:
        if batch_size <= 0 or input_dim <= 0 or intermediate_dim <= 0 or output_dim <= 0:
            return False

        padded_input_dim = ((input_dim + cls.BLOCK_FLOATS - 1) // cls.BLOCK_FLOATS) * cls.BLOCK_FLOATS
        padded_intermediate = ((intermediate_dim + cls.BLOCK_FLOATS - 1) // cls.BLOCK_FLOATS) * cls.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)
        return (
            padded_intermediate <= cls.MAX_INTERMEDIATE_DIM
            and padded_intermediate * padded_input_dim <= cls.MAX_WEIGHT_FLOATS
            and batch_size * padded_input_dim <= cls.MAX_INPUT_FLOATS
            and batch_size * padded_output_dim <= cls.MAX_OUTPUT_FLOATS
        )

    # ── Weight preparation (shared padding logic) ──────────────────────

    def _prepare_weights(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], int]:
        """
        Convert and pad weight tensors to f32 for DPU consumption.
        Returns (gate_f32, up_f32, down_f32, (padded_input_dim, padded_intermediate_dim, padded_output_dim), original_output_dim).
        """
        input_dim = gate_proj.shape[1]
        intermediate_dim = gate_proj.shape[0]
        output_dim = down_proj.shape[0]

        padded_input_dim = ((input_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_intermediate_dim = ((intermediate_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)

        gate_f32 = gate_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        up_f32 = up_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        down_f32 = down_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()

        if padded_input_dim != input_dim:
            padded_gate = torch.zeros(intermediate_dim, padded_input_dim, dtype=torch.float32)
            padded_gate[:, :input_dim] = gate_f32
            gate_f32 = padded_gate

            padded_up = torch.zeros(intermediate_dim, padded_input_dim, dtype=torch.float32)
            padded_up[:, :input_dim] = up_f32
            up_f32 = padded_up

        if padded_intermediate_dim != intermediate_dim:
            padded_gate = torch.zeros(padded_intermediate_dim, gate_f32.shape[1], dtype=torch.float32)
            padded_gate[:intermediate_dim, :] = gate_f32
            gate_f32 = padded_gate

            padded_up = torch.zeros(padded_intermediate_dim, up_f32.shape[1], dtype=torch.float32)
            padded_up[:intermediate_dim, :] = up_f32
            up_f32 = padded_up

            padded_down = torch.zeros(down_f32.shape[0], padded_intermediate_dim, dtype=torch.float32)
            padded_down[:, :intermediate_dim] = down_f32
            down_f32 = padded_down

        if padded_output_dim != output_dim:
            padded_down = torch.zeros(padded_output_dim, down_f32.shape[1], dtype=torch.float32)
            padded_down[:output_dim, :] = down_f32
            down_f32 = padded_down

        return gate_f32, up_f32, down_f32, (padded_input_dim, padded_intermediate_dim, padded_output_dim), output_dim

    # ── NEW: preload — load weights to DPU MRAM if not already resident ─

    def preload(
        self,
        expert_id: int,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> bool:
        """
        Load expert weights to DPU MRAM if not already resident.

        Returns True if weights were actually transferred (cache miss),
        False if they were already resident (cache hit).
        """
        if self._resident_expert_id == expert_id:
            self.preload_hits += 1
            return False

        # Get or create padded f32 tensors (cached to avoid re-padding)
        if expert_id not in self._weight_cache:
            gate_f32, up_f32, down_f32, dims, orig_output_dim = self._prepare_weights(
                gate_proj, up_proj, down_proj
            )
            self._weight_cache[expert_id] = (gate_f32, up_f32, down_f32, dims, orig_output_dim)

        gate_f32, up_f32, down_f32, dims, _orig_output_dim = self._weight_cache[expert_id]
        padded_input_dim, padded_intermediate_dim, padded_output_dim = dims

        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_expert_preload(
            ctypes.c_uint64(expert_id),
            ctypes.c_uint32(padded_input_dim),
            ctypes.c_uint32(padded_intermediate_dim),
            ctypes.c_uint32(padded_output_dim),
            ctypes.c_void_p(gate_f32.data_ptr()),
            ctypes.c_void_p(up_f32.data_ptr()),
            ctypes.c_void_p(down_f32.data_ptr()),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))

        self._resident_expert_id = expert_id
        self.preload_misses += 1
        return True

    # ── NEW: infer — run DPU expert using weights already in MRAM ───────

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the DPU expert kernel using weights already resident in MRAM.
        Only transfers input activations to DPU.
        """
        if self._resident_expert_id == 0:
            raise RuntimeError("No expert weights are resident. Call preload() first.")

        if inputs.ndim != 2:
            raise ValueError("PIMExpertRuntime.infer() expects 2D input tensor.")

        batch_size, input_dim = inputs.shape

        # Pad input if needed
        padded_input_dim = ((input_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        inputs_f32 = inputs.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        if padded_input_dim != input_dim:
            padded_inputs = torch.zeros(batch_size, padded_input_dim, dtype=torch.float32)
            padded_inputs[:, :input_dim] = inputs_f32
            inputs_f32 = padded_inputs

        # Use cached dims from preload
        _, _, _, dims, orig_output_dim = self._weight_cache[self._resident_expert_id]
        _, _, padded_output_dim = dims

        outputs = torch.empty(batch_size, padded_output_dim, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_expert_infer(
            ctypes.c_uint32(batch_size),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))

        return outputs[:, :orig_output_dim].contiguous()

    # ── NEW: evict — clear DPU residency ────────────────────────────────

    def evict(self) -> None:
        """Clear the current DPU weight residency."""
        self._lib.pim_expert_evict()
        self._resident_expert_id = 0

    def evict_cached_weights(self, expert_id: int) -> None:
        """Remove cached pre-padded weights for a specific expert."""
        self._weight_cache.pop(expert_id, None)

    # ── Existing API (backward compatible) ──────────────────────────────

    def expert(
        self,
        inputs: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 2 or gate_proj.ndim != 2 or up_proj.ndim != 2 or down_proj.ndim != 2:
            raise ValueError("PIMExpertRuntime expects 2D tensors.")

        batch_size, input_dim = inputs.shape
        intermediate_dim, gate_input_dim = gate_proj.shape
        up_intermediate_dim, up_input_dim = up_proj.shape
        output_dim, down_input_dim = down_proj.shape
        if input_dim != gate_input_dim or input_dim != up_input_dim:
            raise ValueError("Expert input dimensions do not match projection weights.")
        if intermediate_dim != up_intermediate_dim or intermediate_dim != down_input_dim:
            raise ValueError("Expert intermediate dimensions do not match projection weights.")

        padded_input_dim = ((input_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_intermediate_dim = ((intermediate_dim + self.BLOCK_FLOATS - 1) // self.BLOCK_FLOATS) * self.BLOCK_FLOATS
        padded_output_dim = output_dim + (output_dim % 2)
        if not self.supports_shape(batch_size, input_dim, intermediate_dim, output_dim):
            raise RuntimeError(
                "PIM expert shape unsupported: "
                f"batch={batch_size}, input_dim={input_dim}, intermediate_dim={intermediate_dim}, output_dim={output_dim}"
            )

        inputs_f32 = inputs.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        gate_f32 = gate_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        up_f32 = up_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        down_f32 = down_proj.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()

        if padded_input_dim != input_dim:
            padded_inputs = torch.zeros(batch_size, padded_input_dim, dtype=torch.float32)
            padded_inputs[:, :input_dim] = inputs_f32
            inputs_f32 = padded_inputs

            padded_gate = torch.zeros(intermediate_dim, padded_input_dim, dtype=torch.float32)
            padded_gate[:, :input_dim] = gate_f32
            gate_f32 = padded_gate

            padded_up = torch.zeros(intermediate_dim, padded_input_dim, dtype=torch.float32)
            padded_up[:, :input_dim] = up_f32
            up_f32 = padded_up

        if padded_intermediate_dim != intermediate_dim:
            padded_gate = torch.zeros(padded_intermediate_dim, gate_f32.shape[1], dtype=torch.float32)
            padded_gate[:intermediate_dim, :] = gate_f32
            gate_f32 = padded_gate

            padded_up = torch.zeros(padded_intermediate_dim, up_f32.shape[1], dtype=torch.float32)
            padded_up[:intermediate_dim, :] = up_f32
            up_f32 = padded_up

            padded_down = torch.zeros(down_f32.shape[0], padded_intermediate_dim, dtype=torch.float32)
            padded_down[:, :intermediate_dim] = down_f32
            down_f32 = padded_down

        if padded_output_dim != output_dim:
            padded_down = torch.zeros(padded_output_dim, down_f32.shape[1], dtype=torch.float32)
            padded_down[:output_dim, :] = down_f32
            down_f32 = padded_down

        outputs = torch.empty(batch_size, padded_output_dim, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_expert_run(
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(inputs_f32.shape[1]),
            ctypes.c_uint32(gate_f32.shape[0]),
            ctypes.c_uint32(padded_output_dim),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(gate_f32.data_ptr()),
            ctypes.c_void_p(up_f32.data_ptr()),
            ctypes.c_void_p(down_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        return outputs[:, :output_dim].contiguous()

    # ── Query / diagnostics ─────────────────────────────────────────────

    @property
    def resident_expert_id(self) -> int:
        return self._resident_expert_id

    def last_cycles(self) -> int:
        return int(self._lib.pim_expert_last_cycles())

    def num_dpus(self) -> int:
        return int(self._lib.pim_expert_num_dpus())

    def last_active_dpus(self) -> int:
        return int(self._lib.pim_expert_last_active_dpus())

    def shutdown(self) -> None:
        self._lib.pim_expert_shutdown()
        self._resident_expert_id = 0
        self._weight_cache.clear()
