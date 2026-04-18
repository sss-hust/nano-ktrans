from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .weight_loader import GPTQLinearWeight


@dataclass
class W4A32MatvecResult:
    output: torch.Tensor
    dequantized_weight: torch.Tensor | None


def quantize_symmetric_w4a32(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    bits: int = 4,
    linear_prefix: str = "synthetic",
) -> GPTQLinearWeight:
    if bits != 4:
        raise NotImplementedError("Only 4-bit symmetric quantization is supported.")

    weight_f32 = weight.to(dtype=torch.float32).contiguous()
    output_dim, input_dim = weight_f32.shape
    if input_dim % group_size != 0:
        raise ValueError("input_dim must be divisible by group_size for synthetic W4A32 quantization.")

    num_groups = input_dim // group_size
    zero_point = float(1 << (bits - 1))
    qvalues = torch.empty(output_dim, input_dim, dtype=torch.uint8)
    scales = torch.empty(output_dim, num_groups, dtype=torch.float32)

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        chunk = weight_f32[:, start:end]
        max_abs = chunk.abs().amax(dim=1).clamp(min=1e-6)
        scale = max_abs / ((1 << (bits - 1)) - 1)
        q = torch.round(chunk / scale.unsqueeze(1)).clamp(
            -(1 << (bits - 1)),
            (1 << (bits - 1)) - 1,
        )
        qvalues[:, start:end] = (q + zero_point).to(dtype=torch.uint8)
        scales[:, group_idx] = scale

    packed = pack_int4_matrix(qvalues)
    return GPTQLinearWeight(
        qweight=packed,
        scales=scales,
        zero_points=None,
        group_size=group_size,
        bits=bits,
        sym=True,
        linear_prefix=linear_prefix,
    )


def pack_int4_matrix(qvalues: torch.Tensor) -> torch.Tensor:
    if qvalues.ndim != 2:
        raise ValueError(f"Expected 2D qvalues, got shape={tuple(qvalues.shape)}")
    output_dim, input_dim = qvalues.shape
    if input_dim % 8 != 0:
        raise ValueError("input_dim must be divisible by 8 for 4-bit packing.")

    reshaped = qvalues.to(dtype=torch.int32).reshape(output_dim, input_dim // 8, 8)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).view(1, 1, 8)
    packed = torch.sum(reshaped << shifts, dim=2).to(dtype=torch.int32)
    return packed.contiguous()


def _unpack_packed_words(
    packed_words: torch.Tensor,
    *,
    bits: int,
) -> torch.Tensor:
    values_per_word = 32 // bits
    shifts = torch.arange(values_per_word, dtype=torch.int32, device=packed_words.device) * bits
    unpacked = ((packed_words.to(dtype=torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).to(dtype=torch.float32)
    return unpacked.reshape(packed_words.shape[0], packed_words.shape[1] * values_per_word)


def cpu_w4a32_matvec_grouped(
    inputs: torch.Tensor,
    quantized: GPTQLinearWeight,
) -> W4A32MatvecResult:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D inputs for W4A32 matvec, got shape={tuple(inputs.shape)}")
    if inputs.shape[1] != quantized.input_dim:
        raise ValueError(
            f"Input dim mismatch for W4A32 matvec: inputs.shape={tuple(inputs.shape)}, "
            f"quantized.input_dim={quantized.input_dim}"
        )

    inputs_f32 = inputs.to(dtype=torch.float32)
    batch_size = inputs_f32.shape[0]
    output = torch.zeros(batch_size, quantized.output_dim, dtype=torch.float32)
    values_per_word = quantized.values_per_word
    zero_point_value = float(1 << (quantized.bits - 1))

    for group_idx in range(quantized.num_groups):
        start = group_idx * quantized.group_size
        end = min(start + quantized.group_size, quantized.input_dim)
        start_word = start // values_per_word
        end_word = (end + values_per_word - 1) // values_per_word

        packed = quantized.qweight[:, start_word:end_word]
        qvalues = _unpack_packed_words(packed, bits=quantized.bits)
        offset = start - (start_word * values_per_word)
        qvalues = qvalues[:, offset: offset + (end - start)]

        scale = quantized.scales[:, group_idx].unsqueeze(1)
        if quantized.zero_points is None:
            zero = torch.full_like(scale, zero_point_value)
        else:
            zero = quantized.zero_points[:, group_idx].unsqueeze(1)

        dequant_group = (qvalues - zero) * scale
        output += F.linear(inputs_f32[:, start:end], dequant_group)

    return W4A32MatvecResult(output=output, dequantized_weight=None)


def cpu_w4a32_matvec_dense(
    inputs: torch.Tensor,
    quantized: GPTQLinearWeight,
) -> W4A32MatvecResult:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D inputs for W4A32 matvec, got shape={tuple(inputs.shape)}")
    if inputs.shape[1] != quantized.input_dim:
        raise ValueError(
            f"Input dim mismatch for W4A32 matvec: inputs.shape={tuple(inputs.shape)}, "
            f"quantized.input_dim={quantized.input_dim}"
        )

    weight = quantized.dequantize()
    output = F.linear(inputs.to(dtype=torch.float32), weight)
    return W4A32MatvecResult(output=output, dequantized_weight=weight)


def cpu_w4a32_matvec(
    inputs: torch.Tensor,
    quantized: GPTQLinearWeight,
) -> W4A32MatvecResult:
    return cpu_w4a32_matvec_grouped(inputs, quantized)
