from __future__ import annotations

import shutil
from glob import glob

import pytest
import torch
from safetensors.torch import save_file


def _has_real_dpu() -> bool:
    return bool(glob("/dev/dpu_rank*")) and shutil.which("dpu-upmem-dpurte-clang") is not None


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_linear_runtime_matches_cpu():
    from nano_ktrans.kernels.pim_linear_runtime import PIMLinearRuntime

    torch.manual_seed(0)
    runtime = PIMLinearRuntime.get_shared(rank_count=1)
    inputs = torch.randn(1, 64, dtype=torch.float32)
    weights = torch.randn(128, 64, dtype=torch.float32)

    expected = torch.nn.functional.linear(inputs, weights)
    actual = runtime.linear(inputs, weights)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _has_real_dpu(), reason="Real UPMEM hardware and toolchain are required.")
def test_pim_moe_backend_real_mode_uses_dpu(tmp_path):
    from nano_ktrans.kernels.pim_moe import PIMMoEBackend

    hidden_size = 64
    intermediate_size = 32
    num_experts = 2
    top_k = 2

    weights = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": torch.randn(hidden_size, intermediate_size),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.block_sparse_moe.experts.1.w2.weight": torch.randn(hidden_size, intermediate_size),
    }
    save_file(weights, str(tmp_path / "model.safetensors"))

    backend = PIMMoEBackend(
        layer_idx=0,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gpu_experts_mask=torch.zeros(num_experts, dtype=torch.bool),
        weight_path=str(tmp_path),
        pim_execution_mode="real",
        pim_max_batch_tokens=1,
        pim_kernel_variant="fused",
    )

    hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.long)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    backend.submit_forward(hidden_states, topk_ids, topk_weights, None)
    actual = backend.sync_forward(hidden_states, None).to(dtype=torch.float32)

    ref = torch.zeros(1, hidden_size, dtype=torch.float32)
    for expert_idx in range(num_experts):
        cpu_slot = backend.cpu_expert_lookup[expert_idx]
        gate = torch.nn.functional.linear(hidden_states, backend._gate_proj[cpu_slot].to(dtype=torch.float32))
        up = torch.nn.functional.linear(hidden_states, backend._up_proj[cpu_slot].to(dtype=torch.float32))
        hidden = torch.nn.functional.silu(gate) * up
        expert_output = torch.nn.functional.linear(hidden, backend._down_proj[cpu_slot].to(dtype=torch.float32))
        ref += expert_output * topk_weights[0, expert_idx]

    diagnostics = backend.diagnostics()
    assert diagnostics["runtime_available"] is True
    assert diagnostics["execution_mode"] == "dpu_linear_host_activation"
    assert diagnostics["pim_kernel_variant"] == "fused"
    assert diagnostics["real_dpu_expert_calls"] >= 1
    assert diagnostics["real_dpu_fused_expert_calls"] >= 1
    assert diagnostics["expert_runtime_available"] is True
    assert diagnostics["runtime_dpu_count"] >= 1
    assert torch.allclose(actual, ref, atol=2e-3, rtol=2e-3)
