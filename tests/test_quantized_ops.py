from __future__ import annotations

import torch
from safetensors.torch import save_file


def test_quantize_symmetric_w4a32_roundtrip():
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32

    weight = torch.randn(16, 128, dtype=torch.float32)
    quantized = quantize_symmetric_w4a32(weight, group_size=64, linear_prefix="synthetic")

    dequantized = quantized.dequantize()
    assert dequantized.shape == weight.shape
    assert quantized.qweight.shape == (16, 16)
    assert quantized.scales.shape == (16, 2)
    assert torch.allclose(weight, dequantized, atol=1.0, rtol=0.25)


def test_weight_loader_loads_gptq_expert_linear(tmp_path):
    from nano_ktrans.kernels.quantized_ops import quantize_symmetric_w4a32
    from nano_ktrans.kernels.weight_loader import ExpertWeightLoader

    dense = torch.randn(32, 128, dtype=torch.float32)
    quantized = quantize_symmetric_w4a32(
        dense,
        group_size=64,
        linear_prefix="model.layers.0.mlp.experts.0.gate_proj",
    )
    save_file(
        {
            "model.layers.0.mlp.experts.0.gate_proj.qweight": quantized.qweight.transpose(0, 1).contiguous(),
            "model.layers.0.mlp.experts.0.gate_proj.scales": quantized.scales.transpose(0, 1).contiguous(),
            "model.layers.0.mlp.experts.0.gate_proj.g_idx": (torch.arange(128) // 64).to(dtype=torch.int32),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 4, "group_size": 64, "sym": true, "checkpoint_format": "gptq"}'
    )

    loader = ExpertWeightLoader(str(tmp_path))
    loaded = loader.load_gptq_expert_linear(
        layer_idx=0,
        expert_idx=0,
        proj_name="gate",
    )

    assert torch.equal(loaded.qweight, quantized.qweight)
    assert torch.allclose(loaded.scales, quantized.scales)
    assert loaded.group_size == 64
    assert loaded.sym is True


def test_cpu_w4a32_matvec_matches_linear_from_dequantized_weight():
    from nano_ktrans.kernels.quantized_ops import (
        cpu_w4a32_matvec,
        cpu_w4a32_matvec_dense,
        quantize_symmetric_w4a32,
    )

    weight = torch.randn(24, 128, dtype=torch.float32)
    inputs = torch.randn(3, 128, dtype=torch.float32)
    quantized = quantize_symmetric_w4a32(weight, group_size=64, linear_prefix="synthetic")

    result = cpu_w4a32_matvec(inputs, quantized)
    dense_result = cpu_w4a32_matvec_dense(inputs, quantized)
    expected = torch.nn.functional.linear(inputs, quantized.dequantize())

    assert result.dequantized_weight is None
    assert torch.allclose(result.output, dense_result.output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dense_result.output, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(result.output, expected, atol=1e-5, rtol=1e-5)
