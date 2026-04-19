from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
from nano_ktrans.kernels.quantized_ops import (
    cpu_w4a32_matvec,
    cpu_w4a32_matvec_dense,
    quantize_symmetric_w4a32,
)
from nano_ktrans.kernels.weight_loader import ExpertWeightLoader


def load_quantized_linear_from_model(
    model_path: str,
    *,
    layer_idx: int,
    expert_idx: int,
    proj_name: str,
):
    loader = ExpertWeightLoader(model_path)
    return loader.load_gptq_expert_linear(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        proj_name=proj_name,
    )


def benchmark_cpu(inputs: torch.Tensor, quantized, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        cpu_w4a32_matvec(inputs, quantized)

    outputs = None
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = cpu_w4a32_matvec(inputs, quantized).output
        durations.append(time.perf_counter() - start)

    assert outputs is not None
    return {
        "output": outputs,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
    }


def benchmark_cpu_dense(inputs: torch.Tensor, quantized, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        cpu_w4a32_matvec_dense(inputs, quantized)

    outputs = None
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = cpu_w4a32_matvec_dense(inputs, quantized).output
        durations.append(time.perf_counter() - start)

    assert outputs is not None
    return {
        "output": outputs,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
    }


def benchmark_pim(inputs: torch.Tensor, quantized, repeats: int, warmup: int, rank_count: int, profile: str) -> dict:
    runtime = PIMQuantizedRuntime.get_shared(rank_count=rank_count, profile=profile)
    load_profile = None
    for _ in range(warmup):
        runtime.linear(inputs, quantized)
        load_profile = runtime.last_profile()

    outputs = None
    durations = []
    cycles = []
    input_transfer = []
    launch = []
    output_transfer = []
    runtime_total = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = runtime.linear(inputs, quantized)
        durations.append(time.perf_counter() - start)
        cycles.append(runtime.last_cycles())
        phase_profile = runtime.last_profile()
        input_transfer.append(phase_profile["input_transfer_seconds"])
        launch.append(phase_profile["launch_seconds"])
        output_transfer.append(phase_profile["output_transfer_seconds"])
        runtime_total.append(phase_profile["runtime_total_seconds"])

    assert outputs is not None
    return {
        "output": outputs,
        "load_qweight_transfer_seconds": 0.0 if load_profile is None else load_profile["load_qweight_transfer_seconds"],
        "load_scale_transfer_seconds": 0.0 if load_profile is None else load_profile["load_scale_transfer_seconds"],
        "load_total_seconds": 0.0 if load_profile is None else load_profile["load_total_seconds"],
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
        "kernel_cycles_avg": sum(cycles) / len(cycles),
        "kernel_cycles_min": min(cycles),
        "kernel_cycles_max": max(cycles),
        "runtime_dpu_count": runtime.num_dpus(),
        "input_transfer_seconds_avg": sum(input_transfer) / len(input_transfer),
        "launch_seconds_avg": sum(launch) / len(launch),
        "output_transfer_seconds_avg": sum(output_transfer) / len(output_transfer),
        "runtime_total_seconds_avg": sum(runtime_total) / len(runtime_total),
    }


def benchmark_pim_mode(
    inputs: torch.Tensor,
    quantized,
    repeats: int,
    warmup: int,
    rank_count: int,
    profile: str,
    *,
    kernel_mode: int,
) -> dict:
    runtime = PIMQuantizedRuntime.get_shared(rank_count=rank_count, profile=profile)
    for _ in range(warmup):
        runtime.linear(inputs, quantized, kernel_mode=kernel_mode)

    outputs = None
    durations = []
    cycles = []
    input_transfer = []
    launch = []
    output_transfer = []
    runtime_total = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = runtime.linear(inputs, quantized, kernel_mode=kernel_mode)
        durations.append(time.perf_counter() - start)
        cycles.append(runtime.last_cycles())
        phase_profile = runtime.last_profile()
        input_transfer.append(phase_profile["input_transfer_seconds"])
        launch.append(phase_profile["launch_seconds"])
        output_transfer.append(phase_profile["output_transfer_seconds"])
        runtime_total.append(phase_profile["runtime_total_seconds"])

    assert outputs is not None
    return {
        "output": outputs,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
        "kernel_cycles_avg": sum(cycles) / len(cycles),
        "kernel_cycles_min": min(cycles),
        "kernel_cycles_max": max(cycles),
        "runtime_dpu_count": runtime.num_dpus(),
        "input_transfer_seconds_avg": sum(input_transfer) / len(input_transfer),
        "launch_seconds_avg": sum(launch) / len(launch),
        "output_transfer_seconds_avg": sum(output_transfer) / len(output_transfer),
        "runtime_total_seconds_avg": sum(runtime_total) / len(runtime_total),
    }


def maybe_benchmark_pim_mode(
    inputs: torch.Tensor,
    quantized,
    repeats: int,
    warmup: int,
    rank_count: int,
    profile: str,
    *,
    kernel_mode: int,
) -> dict | None:
    try:
        return benchmark_pim_mode(
            inputs,
            quantized,
            repeats=repeats,
            warmup=warmup,
            rank_count=rank_count,
            profile=profile,
            kernel_mode=kernel_mode,
        )
    except RuntimeError as exc:
        return {"error": str(exc)}


def benchmark_pim_breakdown(inputs: torch.Tensor, quantized, rank_count: int, profile: str) -> dict:
    runtime = PIMQuantizedRuntime.get_shared(rank_count=rank_count, profile=profile)
    runtime.linear(inputs, quantized, kernel_mode=1)
    transfer_only = runtime.last_profile()
    runtime.linear(inputs, quantized, kernel_mode=2)
    unpack_only = runtime.last_profile()
    runtime.linear(inputs, quantized, kernel_mode=3)
    dequant_only = runtime.last_profile()
    runtime.linear(inputs, quantized, kernel_mode=4)
    int8_fixed = runtime.last_profile()
    try:
        runtime.linear(inputs, quantized, kernel_mode=5)
        int8_block_fixed = runtime.last_profile()
    except RuntimeError as exc:
        int8_block_fixed = {"error": str(exc)}
    runtime.linear(inputs, quantized, kernel_mode=0)
    full = runtime.last_profile()
    return {
        "transfer_only_runtime_total_seconds": transfer_only["runtime_total_seconds"],
        "transfer_only_input_transfer_seconds": transfer_only["input_transfer_seconds"],
        "transfer_only_output_transfer_seconds": transfer_only["output_transfer_seconds"],
        "unpack_only_runtime_total_seconds": unpack_only["runtime_total_seconds"],
        "dequant_only_runtime_total_seconds": dequant_only["runtime_total_seconds"],
        "int8_fixed_runtime_total_seconds": int8_fixed["runtime_total_seconds"],
        "int8_block_fixed_runtime_total_seconds": (
            int8_block_fixed["runtime_total_seconds"] if "runtime_total_seconds" in int8_block_fixed else None
        ),
        "int8_block_fixed_error": int8_block_fixed.get("error"),
        "full_runtime_total_seconds": full["runtime_total_seconds"],
        "estimated_compute_seconds": max(
            0.0,
            full["runtime_total_seconds"] - transfer_only["runtime_total_seconds"],
        ),
        "estimated_unpack_seconds": max(
            0.0,
            unpack_only["runtime_total_seconds"] - transfer_only["runtime_total_seconds"],
        ),
        "estimated_dequant_seconds": max(
            0.0,
            dequant_only["runtime_total_seconds"] - unpack_only["runtime_total_seconds"],
        ),
        "estimated_mac_seconds": max(
            0.0,
            full["runtime_total_seconds"] - dequant_only["runtime_total_seconds"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator-only W4A32 CPU vs PIM matvec benchmark.")
    parser.add_argument("--model-path", help="Local path to GPTQ Int4 model.")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--expert-idx", type=int, default=0)
    parser.add_argument("--proj-name", choices=("gate", "up", "down"), default="gate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rank-count", type=int, default=1)
    parser.add_argument("--profile", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic W4A32 weights instead of GPTQ model.")
    parser.add_argument("--synthetic-input-dim", type=int, default=2048)
    parser.add_argument("--synthetic-output-dim", type=int, default=768)
    parser.add_argument("--synthetic-group-size", type=int, default=128)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.synthetic:
        dense = torch.randn(args.synthetic_output_dim, args.synthetic_input_dim, dtype=torch.float32)
        quantized = quantize_symmetric_w4a32(
            dense,
            group_size=args.synthetic_group_size,
            linear_prefix="synthetic",
        )
        source = {
            "kind": "synthetic",
            "input_dim": args.synthetic_input_dim,
            "output_dim": args.synthetic_output_dim,
            "group_size": args.synthetic_group_size,
        }
    else:
        if not args.model_path:
            raise SystemExit("--model-path is required unless --synthetic is used.")
        quantized = load_quantized_linear_from_model(
            args.model_path,
            layer_idx=args.layer_idx,
            expert_idx=args.expert_idx,
            proj_name=args.proj_name,
        )
        source = {
            "kind": "gptq_model",
            "model_path": args.model_path,
            "layer_idx": args.layer_idx,
            "expert_idx": args.expert_idx,
            "proj_name": args.proj_name,
            "linear_prefix": quantized.linear_prefix,
        }

    inputs = torch.randn(args.batch_size, quantized.input_dim, dtype=torch.float32)
    cpu_result = benchmark_cpu(inputs, quantized, repeats=args.repeats, warmup=args.warmup)
    cpu_dense_result = benchmark_cpu_dense(inputs, quantized, repeats=args.repeats, warmup=args.warmup)
    pim_result = benchmark_pim(
        inputs,
        quantized,
        repeats=args.repeats,
        warmup=args.warmup,
        rank_count=args.rank_count,
        profile=args.profile,
    )
    pim_int8_fixed_result = benchmark_pim_mode(
        inputs,
        quantized,
        repeats=args.repeats,
        warmup=args.warmup,
        rank_count=args.rank_count,
        profile=args.profile,
        kernel_mode=4,
    )
    pim_int8_block_fixed_result = maybe_benchmark_pim_mode(
        inputs,
        quantized,
        repeats=args.repeats,
        warmup=args.warmup,
        rank_count=args.rank_count,
        profile=args.profile,
        kernel_mode=5,
    )
    pim_breakdown = benchmark_pim_breakdown(
        inputs,
        quantized,
        rank_count=args.rank_count,
        profile=args.profile,
    )

    diff = (cpu_result["output"] - pim_result["output"]).abs()
    diff_int8_fixed = (cpu_result["output"] - pim_int8_fixed_result["output"]).abs()
    diff_int8_block_fixed = None
    if pim_int8_block_fixed_result is not None and "output" in pim_int8_block_fixed_result:
        diff_int8_block_fixed = (cpu_result["output"] - pim_int8_block_fixed_result["output"]).abs()
    dense_vs_grouped = (cpu_dense_result["output"] - cpu_result["output"]).abs()
    result = {
        "source": source,
        "bits": quantized.bits,
        "group_size": quantized.group_size,
        "sym": quantized.sym,
        "input_dim": quantized.input_dim,
        "output_dim": quantized.output_dim,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cpu_grouped": {k: v for k, v in cpu_result.items() if k != "output"},
        "cpu_dense": {k: v for k, v in cpu_dense_result.items() if k != "output"},
        "pim": {k: v for k, v in pim_result.items() if k != "output"},
        "pim_int8_fixed": {k: v for k, v in pim_int8_fixed_result.items() if k != "output"},
        "pim_int8_block_fixed": (
            {k: v for k, v in pim_int8_block_fixed_result.items() if k != "output"}
            if pim_int8_block_fixed_result is not None
            else None
        ),
        "pim_breakdown": pim_breakdown,
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_abs_error_pim_int8_fixed": float(diff_int8_fixed.max().item()),
        "mean_abs_error_pim_int8_fixed": float(diff_int8_fixed.mean().item()),
        "max_abs_error_pim_int8_block_fixed": (
            float(diff_int8_block_fixed.max().item()) if diff_int8_block_fixed is not None else None
        ),
        "mean_abs_error_pim_int8_block_fixed": (
            float(diff_int8_block_fixed.mean().item()) if diff_int8_block_fixed is not None else None
        ),
        "max_abs_error_cpu_dense_vs_grouped": float(dense_vs_grouped.max().item()),
        "speedup_pim_vs_cpu_grouped": (
            cpu_result["seconds_avg"] / pim_result["seconds_avg"] if pim_result["seconds_avg"] > 0 else None
        ),
        "speedup_pim_vs_cpu_dense": (
            cpu_dense_result["seconds_avg"] / pim_result["seconds_avg"] if pim_result["seconds_avg"] > 0 else None
        ),
        "speedup_pim_int8_fixed_vs_cpu_grouped": (
            cpu_result["seconds_avg"] / pim_int8_fixed_result["seconds_avg"]
            if pim_int8_fixed_result["seconds_avg"] > 0
            else None
        ),
        "speedup_pim_int8_fixed_vs_cpu_dense": (
            cpu_dense_result["seconds_avg"] / pim_int8_fixed_result["seconds_avg"]
            if pim_int8_fixed_result["seconds_avg"] > 0
            else None
        ),
        "speedup_pim_int8_block_fixed_vs_cpu_grouped": (
            cpu_result["seconds_avg"] / pim_int8_block_fixed_result["seconds_avg"]
            if pim_int8_block_fixed_result is not None
            and "seconds_avg" in pim_int8_block_fixed_result
            and pim_int8_block_fixed_result["seconds_avg"] > 0
            else None
        ),
        "speedup_pim_int8_block_fixed_vs_cpu_dense": (
            cpu_dense_result["seconds_avg"] / pim_int8_block_fixed_result["seconds_avg"]
            if pim_int8_block_fixed_result is not None
            and "seconds_avg" in pim_int8_block_fixed_result
            and pim_int8_block_fixed_result["seconds_avg"] > 0
            else None
        ),
        "speedup_pim_int8_fixed_vs_pim_full": (
            pim_result["seconds_avg"] / pim_int8_fixed_result["seconds_avg"]
            if pim_int8_fixed_result["seconds_avg"] > 0
            else None
        ),
        "speedup_pim_int8_block_fixed_vs_pim_full": (
            pim_result["seconds_avg"] / pim_int8_block_fixed_result["seconds_avg"]
            if pim_int8_block_fixed_result is not None
            and "seconds_avg" in pim_int8_block_fixed_result
            and pim_int8_block_fixed_result["seconds_avg"] > 0
            else None
        ),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)


if __name__ == "__main__":
    main()
