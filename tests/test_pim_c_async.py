"""ADR-002 M-24 Stage A tests: C-level async submit of the fused op.

These tests validate that ``pim_quantized_run_many_fused_silu_async`` +
``pim_quantized_fused_wait`` produce the same output as the synchronous
fused path, while letting Python proceed between submit and wait.

Tests gated on real UPMEM hardware (same gate as other DPU tests).
"""
from __future__ import annotations

import shutil
from glob import glob

import pytest
import torch


def _has_real_dpu() -> bool:
    return bool(glob("/dev/dpu_rank*")) and shutil.which("dpu-upmem-dpurte-clang") is not None


# ── Signature / binding ────────────────────────────────────────────────────


def test_pim_c_async_api_surface():
    from nano_ktrans.kernels.pim_quantized_runtime import (
        PIMQuantizedRuntime, PIMFusedAsyncHandle,
    )
    assert hasattr(PIMQuantizedRuntime, "submit_many_fused_silu_async")
    assert hasattr(PIMFusedAsyncHandle, "wait")


def test_pim_c_async_bridge_symbols_linked():
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    import os, ctypes
    if not os.path.exists("/dev/dpu_rank0"):
        pytest.skip("no DPU device; need _lib instance")
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_async_sig")
    assert hasattr(rt._lib, "pim_quantized_run_many_fused_silu_async")
    assert hasattr(rt._lib, "pim_quantized_fused_wait")
    assert rt._lib.pim_quantized_run_many_fused_silu_async.restype == ctypes.c_int
    assert rt._lib.pim_quantized_fused_wait.restype == ctypes.c_int


# ── Numerical equivalence: async vs sync fused ─────────────────────────────


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware required.")
def test_pim_c_async_matches_sync_fused_one_expert():
    """Core Stage A correctness gate: async fused == sync fused bit-exact.

    Because both paths call the exact same C function
    (pim_quantized_run_many_fused_silu), just on different threads, the
    numerical output must be bit-identical (unlike Stage B vs legacy
    which had an expf vs torch.sigmoid ulp diff).
    """
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    torch.manual_seed(111)
    hidden = 128
    inter = 128
    gate = quantize_symmetric_w4a32(
        torch.randn(inter, hidden), group_size=64, linear_prefix="g_async",
    )
    up = quantize_symmetric_w4a32(
        torch.randn(inter, hidden), group_size=64, linear_prefix="u_async",
    )
    down = quantize_symmetric_w4a32(
        torch.randn(hidden, inter), group_size=64, linear_prefix="d_async",
    )
    inputs = torch.randn(1, hidden)

    rt_gu = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_async_eq_gu")
    rt_dn = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_async_eq_dn")

    # Sync fused run
    gu_slot, gu_pin, gu_concat, gc, uc = \
        rt_gu.preload_concat_and_get_slot(60001, gate, up, kernel_mode=4)
    dn_slot, dn_pin, dn_pout, dn_orig = \
        rt_dn.preload_and_get_slot(60002, down, kernel_mode=4)
    sync_out = PIMQuantizedRuntime.infer_many_fused_silu(
        rt_gu, rt_dn,
        [(inputs, gu_slot, gu_pin, gu_concat, gc, uc, dn_slot, dn_pin, dn_pout)],
    )[0][:, :dn_orig].contiguous()

    # Async fused run (preloads are cache hits, so no weight DMA)
    gu_slot2, gu_pin2, gu_concat2, gc2, uc2 = \
        rt_gu.preload_concat_and_get_slot(60001, gate, up, kernel_mode=4)
    dn_slot2, dn_pin2, dn_pout2, dn_orig2 = \
        rt_dn.preload_and_get_slot(60002, down, kernel_mode=4)
    handle = PIMQuantizedRuntime.submit_many_fused_silu_async(
        rt_gu, rt_dn,
        [(inputs, gu_slot2, gu_pin2, gu_concat2, gc2, uc2,
          dn_slot2, dn_pin2, dn_pout2)],
    )
    # Simulate GPU-side work window.  On real hw this is where HybridMoE
    # would run its GPU expert loop.  Keep it short to avoid making the
    # test slow; the point is only that wait() works under any delay.
    import time
    time.sleep(0.002)
    async_outputs = handle.wait()
    async_out = async_outputs[0][:, :dn_orig2].contiguous()

    # Same C function, same data, different threads → bit-exact expected.
    assert torch.equal(sync_out, async_out), (
        f"sync vs async differ: max_err={(sync_out-async_out).abs().max()}"
    )


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware required.")
def test_pim_c_async_double_wait_raises():
    """wait() called twice on the same handle must raise, not double-join."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    torch.manual_seed(121)
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_async_dw")
    gate = quantize_symmetric_w4a32(torch.randn(128, 128), group_size=64, linear_prefix="gd")
    up = quantize_symmetric_w4a32(torch.randn(128, 128), group_size=64, linear_prefix="ud")
    down = quantize_symmetric_w4a32(torch.randn(128, 128), group_size=64, linear_prefix="dd")

    gu_slot, gu_pin, gu_concat, gc, uc = \
        rt.preload_concat_and_get_slot(70001, gate, up, kernel_mode=4)
    dn_slot, dn_pin, dn_pout, dn_orig = \
        rt.preload_and_get_slot(70002, down, kernel_mode=4)
    handle = PIMQuantizedRuntime.submit_many_fused_silu_async(
        rt, rt,
        [(torch.randn(1, 128), gu_slot, gu_pin, gu_concat, gc, uc,
          dn_slot, dn_pin, dn_pout)],
    )
    _ = handle.wait()
    with pytest.raises(RuntimeError, match="wait.*called twice"):
        handle.wait()


def test_pim_c_async_empty_submit_returns_empty_handle():
    """Zero-request submit must produce a wait()-able handle returning []."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    import os
    if not os.path.exists("/dev/dpu_rank0"):
        pytest.skip("no DPU device")
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_async_empty")
    handle = PIMQuantizedRuntime.submit_many_fused_silu_async(rt, rt, [])
    out = handle.wait()
    assert out == []


# ── Backend-level: enable_c_async_submit end-to-end ────────────────────────


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware required.")
def test_pim_moe_backend_c_async_matches_legacy(tmp_path):
    """End-to-end backend: toggling enable_c_async_submit must not change
    MoE output vs the legacy synchronous path."""
    from safetensors.torch import save_file
    from nano_ktrans.kernels.pim_moe import PIMMoEBackend
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32
    from nano_ktrans.utils.context import reset_context, set_context

    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    top_k = 2

    dummy_weights = {
        f"model.layers.0.block_sparse_moe.experts.{e}.{proj}.weight":
            torch.randn(
                intermediate_size if proj in ("w1", "w3") else hidden_size,
                hidden_size if proj in ("w1", "w3") else intermediate_size,
            )
        for e in range(num_experts)
        for proj in ("w1", "w2", "w3")
    }
    save_file(dummy_weights, str(tmp_path / "model.safetensors"))

    torch.manual_seed(331)
    gptq_experts = {}
    for slot in range(num_experts):
        gptq_experts[slot] = {
            proj: quantize_symmetric_w4a32(
                torch.randn(
                    intermediate_size if proj in ("gate", "up") else hidden_size,
                    hidden_size if proj in ("gate", "up") else intermediate_size,
                ),
                group_size=64, linear_prefix=f"{proj}_async_{slot}",
            )
            for proj in ("gate", "up", "down")
        }

    def _make(enable_c_async: bool) -> PIMMoEBackend:
        be = PIMMoEBackend(
            layer_idx=0,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gpu_experts_mask=torch.zeros(num_experts, dtype=torch.bool),
            weight_path=str(tmp_path),
            pim_execution_mode="real",
            pim_max_batch_tokens=1,
            pim_kernel_variant="linear",
            enable_cost_model_routing=False,
            enable_c_async_submit=enable_c_async,
        )
        be.is_gptq = True
        be._gptq_experts = gptq_experts
        be.cpu_expert_lookup = {e: e for e in range(num_experts)}
        gu_rt, dn_rt = be._try_init_quantized_runtimes_dual()
        be.quantized_runtime = gu_rt
        be.quantized_runtime_down = dn_rt
        return be

    hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.long)
    topk_weights = torch.tensor([[0.55, 0.45]], dtype=torch.float32)

    reset_context()
    set_context(is_prefill=False)
    try:
        be_legacy = _make(enable_c_async=False)
        be_legacy.submit_forward(hidden_states, topk_ids, topk_weights, None)
        out_legacy = be_legacy.sync_forward(hidden_states, None).to(dtype=torch.float32).clone()
        diag_legacy = be_legacy.diagnostics()
    finally:
        reset_context()

    reset_context()
    set_context(is_prefill=False)
    try:
        be_async = _make(enable_c_async=True)
        be_async.submit_forward(hidden_states, topk_ids, topk_weights, None)
        out_async = be_async.sync_forward(hidden_states, None).to(dtype=torch.float32).clone()
        diag_async = be_async.diagnostics()
    finally:
        reset_context()

    # PIM-participation guard: both paths must use the DPU.
    assert diag_legacy["real_dpu_expert_calls"] > 0, "legacy didn't hit PIM"
    assert diag_async["real_dpu_expert_calls"] > 0, "c_async didn't hit PIM"
    # Async path must record at least one submit.
    assert diag_async["enable_c_async_submit"] is True
    assert diag_async["c_async_submit_count"] >= 1
    assert diag_async["c_async_sync_wait_seconds_count"] >= 1
    # Legacy path had no c_async submits.
    assert diag_legacy["enable_c_async_submit"] is False
    assert diag_legacy["c_async_submit_count"] == 0
    # Numerical match within PIM int8 tolerance.
    max_err = (out_async - out_legacy).abs().max().item()
    assert max_err < 5.0, f"async vs legacy max_abs_err={max_err:.4f}"
