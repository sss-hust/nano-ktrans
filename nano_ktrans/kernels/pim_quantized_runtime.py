from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
import threading
from pathlib import Path

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

    _shared: dict[tuple[str, int], "PIMQuantizedRuntime"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, *, profile: str = "", rank_count: int = 1) -> None:
        self.profile = profile
        self.rank_count = rank_count
        native_dir = Path(__file__).resolve().parent / "pim_native"
        build_dir = native_dir / "build"
        self._build_if_needed(native_dir, build_dir)

        self.kernel_path = build_dir / "pim_quantized_kernel"
        self.lib_path = build_dir / "libpim_quantized_bridge.so"
        if not self.kernel_path.exists() or not self.lib_path.exists():
            raise RuntimeError("PIM quantized bridge build did not produce the expected artifacts.")

        self._lib = ctypes.CDLL(str(self.lib_path))
        self._lib.pim_quantized_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_init.restype = ctypes.c_int
        self._lib.pim_quantized_load_weights.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_load_weights.restype = ctypes.c_int
        self._lib.pim_quantized_run.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.pim_quantized_run.restype = ctypes.c_int
        self._lib.pim_quantized_last_cycles.argtypes = []
        self._lib.pim_quantized_last_cycles.restype = ctypes.c_uint64
        self._lib.pim_quantized_last_input_transfer_seconds.argtypes = []
        self._lib.pim_quantized_last_input_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_launch_seconds.argtypes = []
        self._lib.pim_quantized_last_launch_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_output_transfer_seconds.argtypes = []
        self._lib.pim_quantized_last_output_transfer_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_last_total_seconds.argtypes = []
        self._lib.pim_quantized_last_total_seconds.restype = ctypes.c_double
        self._lib.pim_quantized_num_dpus.argtypes = []
        self._lib.pim_quantized_num_dpus.restype = ctypes.c_uint32
        self._lib.pim_quantized_shutdown.argtypes = []
        self._lib.pim_quantized_shutdown.restype = None

        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_init(
            os.fsencode(str(self.kernel_path)),
            profile.encode() if profile else None,
            ctypes.c_uint32(rank_count),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        self._loaded_signature: tuple[int, int, int] | None = None

    @classmethod
    def get_shared(cls, *, profile: str = "", rank_count: int = 1) -> "PIMQuantizedRuntime":
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
        kernel_path = build_dir / "pim_quantized_kernel"
        lib_path = build_dir / "libpim_quantized_bridge.so"
        sources = [
            native_dir / "build.sh",
            native_dir / "dpu_quantized_kernel.c",
            native_dir / "host_quantized_bridge.c",
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

    def linear(self, inputs: torch.Tensor, quantized: GPTQLinearWeight) -> torch.Tensor:
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

        signature = (padded_input_dim, padded_output_dim, quantized.group_size)
        if self._loaded_signature != signature:
            error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
            rc = self._lib.pim_quantized_load_weights(
                ctypes.c_uint32(padded_input_dim),
                ctypes.c_uint32(padded_output_dim),
                ctypes.c_uint32(quantized.group_size),
                ctypes.c_void_p(qweight_i32.data_ptr()),
                ctypes.c_void_p(scales_f32.data_ptr()),
                error_buffer,
                len(error_buffer),
            )
            if rc != 0:
                raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
            self._loaded_signature = signature

        outputs = torch.empty(batch_size, padded_output_dim, dtype=torch.float32)
        error_buffer = ctypes.create_string_buffer(self.ERROR_BUFFER_SIZE)
        rc = self._lib.pim_quantized_run(
            ctypes.c_uint32(batch_size),
            ctypes.c_void_p(inputs_f32.data_ptr()),
            ctypes.c_void_p(outputs.data_ptr()),
            error_buffer,
            len(error_buffer),
        )
        if rc != 0:
            raise RuntimeError(error_buffer.value.decode("utf-8", errors="replace"))
        return outputs[:, :output_dim].contiguous()

    def last_cycles(self) -> int:
        return int(self._lib.pim_quantized_last_cycles())

    def last_profile(self) -> dict[str, float]:
        return {
            "input_transfer_seconds": float(self._lib.pim_quantized_last_input_transfer_seconds()),
            "launch_seconds": float(self._lib.pim_quantized_last_launch_seconds()),
            "output_transfer_seconds": float(self._lib.pim_quantized_last_output_transfer_seconds()),
            "runtime_total_seconds": float(self._lib.pim_quantized_last_total_seconds()),
        }

    def num_dpus(self) -> int:
        return int(self._lib.pim_quantized_num_dpus())

    def shutdown(self) -> None:
        self._lib.pim_quantized_shutdown()
        self._loaded_signature = None
