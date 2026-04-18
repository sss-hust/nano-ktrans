from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
from nano_ktrans.kernels.quantized_ops import cpu_w4a32_matvec, quantize_symmetric_w4a32
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


def benchmark_pim(inputs: torch.Tensor, quantized, repeats: int, warmup: int, rank_count: int, profile: str) -> dict:
    runtime = PIMQuantizedRuntime.get_shared(rank_count=rank_count, profile=profile)
    for _ in range(warmup):
        runtime.linear(inputs, quantized)

    outputs = None
    durations = []
    cycles = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = runtime.linear(inputs, quantized)
        durations.append(time.perf_counter() - start)
        cycles.append(runtime.last_cycles())

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
    pim_result = benchmark_pim(
        inputs,
        quantized,
        repeats=args.repeats,
        warmup=args.warmup,
        rank_count=args.rank_count,
        profile=args.profile,
    )

    diff = (cpu_result["output"] - pim_result["output"]).abs()
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
        "cpu": {k: v for k, v in cpu_result.items() if k != "output"},
        "pim": {k: v for k, v in pim_result.items() if k != "output"},
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "speedup_pim_vs_cpu": (
            cpu_result["seconds_avg"] / pim_result["seconds_avg"] if pim_result["seconds_avg"] > 0 else None
        ),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)


if __name__ == "__main__":
    main()
