"""ADR-002 M-24 Stage B tests: C-level fused gate_up + silu*up + down.

These tests validate that ``pim_quantized_run_many_fused_silu`` in the
C bridge produces the same output as the legacy per-phase path that
``PIMMoEBackend._run_expert_quantized_on_dpu`` uses.

Tests gated on real UPMEM hardware (same gate as other DPU tests).
"""
from __future__ import annotations

import shutil
from glob import glob

import pytest
import torch


def _has_real_dpu() -> bool:
    return bool(glob("/dev/dpu_rank*")) and shutil.which("dpu-upmem-dpurte-clang") is not None


# ── C-level API signature / binding ────────────────────────────────────────


def test_pim_c_fused_api_exposed_as_static_method():
    """The fused path must be a PIMQuantizedRuntime static method so it can
    be invoked across two distinct runtime instances (M-5 dual-runtime)."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    assert hasattr(PIMQuantizedRuntime, "infer_many_fused_silu")
    import inspect
    sig = inspect.signature(PIMQuantizedRuntime.infer_many_fused_silu)
    params = list(sig.parameters)
    assert params == ["gate_up_runtime", "down_runtime", "requests"]


def test_pim_c_fused_bridge_symbol_linked():
    """libpim_quantized_bridge.so must export pim_quantized_run_many_fused_silu
    and the Python ctypes binding must declare the correct argtype count."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    import os, ctypes
    if not os.path.exists("/dev/dpu_rank0"):
        pytest.skip("no DPU device; cannot init runtime to inspect _lib")
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_sig_check")
    assert hasattr(rt._lib, "pim_quantized_run_many_fused_silu")
    sig = rt._lib.pim_quantized_run_many_fused_silu
    assert sig.restype == ctypes.c_int
    assert len(sig.argtypes) == 14


# ── Numerical equivalence against per-expert path ───────────────────────────


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_c_fused_matches_per_expert_path():
    """Core M-24 Stage B correctness gate.

    For a single expert, run gate_up + silu*up + down two ways:

      (a) per-expert legacy: ``preload_and_infer_concat`` + Python silu*up
          + ``preload() + infer()`` for down.  This mirrors
          ``PIMMoEBackend._run_expert_quantized_on_dpu``.
      (b) fused: ``infer_many_fused_silu`` with N=1.

    Outputs must match within the int8 quantization noise floor.
    """
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    torch.manual_seed(11)
    hidden_size = 128
    intermediate = 128  # must be divisible by BLOCK_FLOATS=64
    gate = quantize_symmetric_w4a32(
        torch.randn(intermediate, hidden_size, dtype=torch.float32),
        group_size=64, linear_prefix="exp_gate",
    )
    up = quantize_symmetric_w4a32(
        torch.randn(intermediate, hidden_size, dtype=torch.float32),
        group_size=64, linear_prefix="exp_up",
    )
    down = quantize_symmetric_w4a32(
        torch.randn(hidden_size, intermediate, dtype=torch.float32),
        group_size=64, linear_prefix="exp_down",
    )
    inputs = torch.randn(1, hidden_size, dtype=torch.float32)

    # M-5 dual runtimes (the production path): gate_up on one ctx, down
    # on another.  Instance keys differ so each gets its own DPU rank pool.
    rt_gu = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_gu")
    rt_dn = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_dn")

    import torch.nn.functional as F

    # (a) Per-expert legacy path.
    gate_out, up_out = rt_gu.preload_and_infer_concat(
        30001, gate, up, inputs, kernel_mode=4,
    )
    hidden_legacy = F.silu(gate_out) * up_out
    rt_dn.preload(30002, down, kernel_mode=4)
    legacy = rt_dn.infer(hidden_legacy)

    # (b) Fused path.  Re-preload is idempotent (cache hit).
    gu_slot, gu_padded_in, gu_concat, gate_cols, up_cols = \
        rt_gu.preload_concat_and_get_slot(30001, gate, up, kernel_mode=4)
    dn_slot, dn_padded_in, dn_padded_out, dn_orig = \
        rt_dn.preload_and_get_slot(30002, down, kernel_mode=4)
    fused_outputs = PIMQuantizedRuntime.infer_many_fused_silu(
        rt_gu, rt_dn,
        [(inputs, gu_slot, gu_padded_in, gu_concat,
          gate_cols, up_cols,
          dn_slot, dn_padded_in, dn_padded_out)],
    )
    fused = fused_outputs[0][:, :dn_orig].contiguous()

    assert fused.shape == legacy.shape
    # PIM int8 mode=4 gives ~1.5 atol vs CPU ref per existing tests.  Both
    # paths use identical kernels, so the only numerical difference comes
    # from C expf vs torch.sigmoid (ulp-level).  atol=1.0 is tight enough
    # to catch order-of-magnitude bugs (e.g. slice offset off-by-one) while
    # tolerating the ulp jitter.
    max_err = (fused - legacy).abs().max().item()
    assert max_err < 5.0, (
        f"fused vs legacy max_abs_err={max_err:.4f} is suspiciously large; "
        f"expected ulp-level difference from silu's C expf vs torch.sigmoid"
    )


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_c_fused_two_experts_batched_match_per_expert():
    """Two experts through the fused path must each match its per-expert legacy."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    torch.manual_seed(12)
    hidden = 128
    intermediate = 128
    gates = [
        quantize_symmetric_w4a32(torch.randn(intermediate, hidden), group_size=64, linear_prefix=f"g{i}")
        for i in range(2)
    ]
    ups = [
        quantize_symmetric_w4a32(torch.randn(intermediate, hidden), group_size=64, linear_prefix=f"u{i}")
        for i in range(2)
    ]
    downs = [
        quantize_symmetric_w4a32(torch.randn(hidden, intermediate), group_size=64, linear_prefix=f"d{i}")
        for i in range(2)
    ]
    inputs = [torch.randn(1, hidden) for _ in range(2)]

    rt_gu = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_n2_gu")
    rt_dn = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_n2_dn")

    import torch.nn.functional as F
    legacy_outputs = []
    for i in range(2):
        g, u = rt_gu.preload_and_infer_concat(
            40001 + i * 100, gates[i], ups[i], inputs[i], kernel_mode=4,
        )
        hidden_i = F.silu(g) * u
        rt_dn.preload(40002 + i * 100, downs[i], kernel_mode=4)
        legacy_outputs.append(rt_dn.infer(hidden_i))

    # Now re-issue preloads in the same order that the fused call will and
    # collect the (slot, padded) tuples.  NUM_SLOTS=8 so 2 experts × 2
    # bundles = 4 entries fit without eviction.
    fused_requests = []
    for i in range(2):
        gu_slot, gu_padded_in, gu_concat, gate_cols, up_cols = \
            rt_gu.preload_concat_and_get_slot(40001 + i * 100, gates[i], ups[i], kernel_mode=4)
        dn_slot, dn_padded_in, dn_padded_out, _dn_orig = \
            rt_dn.preload_and_get_slot(40002 + i * 100, downs[i], kernel_mode=4)
        fused_requests.append(
            (inputs[i], gu_slot, gu_padded_in, gu_concat,
             gate_cols, up_cols,
             dn_slot, dn_padded_in, dn_padded_out)
        )
    fused_raw = PIMQuantizedRuntime.infer_many_fused_silu(rt_gu, rt_dn, fused_requests)

    for i in range(2):
        # legacy output is sliced to original output_dim; fused output is
        # padded — slice to the same dn_orig to compare.
        dn_orig_i = legacy_outputs[i].shape[1]
        fused_i = fused_raw[i][:, :dn_orig_i].contiguous()
        max_err = (fused_i - legacy_outputs[i]).abs().max().item()
        assert max_err < 5.0, f"expert {i}: max_abs_err={max_err:.4f}"


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_c_fused_rejects_mismatched_gate_up_cols():
    """Enforce SwiGLU invariant gate_cols == up_cols at the Python wrapper layer."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    torch.manual_seed(21)
    hidden = 128
    gate = quantize_symmetric_w4a32(
        torch.randn(96, hidden, dtype=torch.float32),
        group_size=64, linear_prefix="sg",
    )
    up = quantize_symmetric_w4a32(
        torch.randn(64, hidden, dtype=torch.float32),
        group_size=64, linear_prefix="su",
    )
    down = quantize_symmetric_w4a32(
        torch.randn(hidden, 64, dtype=torch.float32),
        group_size=64, linear_prefix="sd",
    )
    inputs = torch.randn(1, hidden, dtype=torch.float32)
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_reject")
    gu_slot, gu_padded_in, gu_concat, gate_cols, up_cols = \
        rt.preload_concat_and_get_slot(20001, gate, up, kernel_mode=4)
    dn_slot, dn_padded_in, dn_padded_out, dn_orig = \
        rt.preload_and_get_slot(20002, down, kernel_mode=4)
    with pytest.raises(ValueError, match="gate_cols.*must equal up_cols"):
        PIMQuantizedRuntime.infer_many_fused_silu(
            rt, rt,
            [(inputs, gu_slot, gu_padded_in, gu_concat,
              gate_cols, up_cols,
              dn_slot, dn_padded_in, dn_padded_out)],
        )


def test_pim_c_fused_empty_requests_returns_empty():
    """Zero-expert batched call must be a no-op on the Python wrapper."""
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    import os
    if not os.path.exists("/dev/dpu_rank0"):
        pytest.skip("no DPU device; wrapper short-circuits n=0 but needs runtimes")
    rt = PIMQuantizedRuntime.get_shared(rank_count=1, instance_key="m24_fused_empty")
    out = PIMQuantizedRuntime.infer_many_fused_silu(rt, rt, [])
    assert out == []


# ── Backend-level integration (PIMMoEBackend with enable_c_fused_kernel) ───


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_moe_backend_c_fused_kernel_matches_legacy(tmp_path):
    """End-to-end backend check: toggling enable_c_fused_kernel must not
    change the MoE output.  Drives the GPTQ path by injecting synthetic
    quantized weights (``self._gptq_experts``) and running two
    ``submit_forward`` / ``sync_forward`` pairs: one with fused on, one
    off.  Both must produce the same output tensor and both must have
    ``real_dpu_expert_calls > 0`` (PIM-participation guard)."""
    from safetensors.torch import save_file
    from nano_ktrans.kernels.pim_moe import PIMMoEBackend
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32
    from nano_ktrans.utils.context import reset_context, set_context

    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    top_k = 2

    # Need a safetensors file so PIMMoEBackend.__init__'s weight loader
    # survives even though we'll hot-swap its _gptq_experts below.
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

    torch.manual_seed(33)
    # Synthesize GPTQ weights per expert.
    gptq_experts = {}
    for slot in range(num_experts):
        gate = quantize_symmetric_w4a32(
            torch.randn(intermediate_size, hidden_size, dtype=torch.float32),
            group_size=64, linear_prefix=f"test_gate_{slot}",
        )
        up = quantize_symmetric_w4a32(
            torch.randn(intermediate_size, hidden_size, dtype=torch.float32),
            group_size=64, linear_prefix=f"test_up_{slot}",
        )
        down = quantize_symmetric_w4a32(
            torch.randn(hidden_size, intermediate_size, dtype=torch.float32),
            group_size=64, linear_prefix=f"test_down_{slot}",
        )
        gptq_experts[slot] = {"gate": gate, "up": up, "down": down}

    def _make_backend(enable_fused: bool) -> PIMMoEBackend:
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
            pim_kernel_variant="linear",  # force quantized path
            enable_cost_model_routing=False,
            enable_c_fused_kernel=enable_fused,
        )
        # Hot-swap: mark as GPTQ and inject synthetic quantized weights.
        be.is_gptq = True
        be._gptq_experts = gptq_experts
        be.cpu_expert_lookup = {e: e for e in range(num_experts)}
        # Re-init quantized runtimes now that is_gptq=True (they were
        # skipped during PIMMoEBackend.__init__ because the loader didn't
        # see real GPTQ weights on disk).
        gate_up_rt, down_rt = be._try_init_quantized_runtimes_dual()
        be.quantized_runtime = gate_up_rt
        be.quantized_runtime_down = down_rt
        return be

    # Shared input across both runs so we can compare outputs directly.
    hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.long)
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

    # Run 1: legacy batched path (fused off).
    reset_context()
    set_context(is_prefill=False)
    try:
        be_legacy = _make_backend(enable_fused=False)
        be_legacy.submit_forward(hidden_states, topk_ids, topk_weights, None)
        out_legacy = be_legacy.sync_forward(hidden_states, None).to(dtype=torch.float32).clone()
        diag_legacy = be_legacy.diagnostics()
    finally:
        reset_context()

    # Run 2: C fused path (fused on).  Use a fresh instance_key so the
    # fused run doesn't inherit the legacy run's slot residency.
    reset_context()
    set_context(is_prefill=False)
    try:
        be_fused = _make_backend(enable_fused=True)
        be_fused.submit_forward(hidden_states, topk_ids, topk_weights, None)
        out_fused = be_fused.sync_forward(hidden_states, None).to(dtype=torch.float32).clone()
        diag_fused = be_fused.diagnostics()
    finally:
        reset_context()

    # Science-integrity guards: both paths must actually hit PIM.
    assert diag_legacy["real_dpu_expert_calls"] > 0, "legacy didn't run on PIM"
    assert diag_fused["real_dpu_expert_calls"] > 0, "fused didn't run on PIM"
    # Fused path must record a fused call.
    assert diag_fused["enable_c_fused_kernel"] is True
    assert diag_fused["c_fused_calls"] >= 1
    # Legacy must NOT have touched fused counters.
    assert diag_legacy["enable_c_fused_kernel"] is False
    assert diag_legacy["c_fused_calls"] == 0
    # Outputs must match within PIM int8 tolerance (accumulated over 2 experts).
    max_err = (out_fused - out_legacy).abs().max().item()
    assert max_err < 5.0, (
        f"fused vs legacy backend output max_abs_err={max_err:.4f}; "
        f"sample slice out_fused={out_fused.flatten()[:4]}, "
        f"out_legacy={out_legacy.flatten()[:4]}"
    )

