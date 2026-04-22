#!/usr/bin/env python3
"""
M1-T2 — PIM operator-only sweep across {shape, batch, rank, kernel_mode}.

Produces the canonical baseline dataset referenced by ADR-002 §5.1 Level-2.
Every milestone (M-2 real T-MAC, M-3 cost model, M-4 研究增量) must be judged
against deltas on top of the table this script generates.

Scope
-----
* Real UPMEM DPU only (pim_quantized_runtime).  Simulator runs are rejected by
  ADR-002 §5.3; this script simply fails closed if DPU cannot be allocated.
* Real ``Qwen/Qwen3-30B-A3B-GPTQ-Int4`` expert projections by default.
  Falls back to the synthetic quantizer only when explicitly requested via
  ``--synthetic`` (for smoke-running on machines that lack the checkpoint).

Grid (defaults match ADR-002 §2):
    shapes   = 3   (gate, up, down)
    batches  = [1, 2, 4, 8]
    ranks    = [1, 4, 8, 16, 32]
    kmodes   = [3, 4, 6]          # full soft-float, int8 fixed, "current T-MAC"
    warmup=3, repeats=10

Output
------
JSON with one ``results`` entry per (shape, batch, rank, kernel_mode) cell,
including ``seconds_{avg,min,max}``, ``launch/transfer`` breakdown,
``max_abs_error`` vs CPU grouped baseline, and ``pim_vs_cpu_grouped_ratio``.
Each row is self-describing so downstream sweeps can diff without re-running
CPU baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
from nano_ktrans.kernels.quantized_ops import (
    cpu_w4a32_matvec,
    cpu_w4a32_matvec_dense,
    quantize_symmetric_w4a32,
)
from nano_ktrans.kernels.weight_loader import ExpertWeightLoader, GPTQLinearWeight


# ---------------------------------------------------------------------------
#  Shape catalog
# ---------------------------------------------------------------------------

#: Qwen3-30B-A3B MoE expert projection shapes.
#: hidden_size=2048, moe_intermediate_size=768.
#:
#: Naming convention: (proj_name, in_features, out_features).
#: `gate` / `up` go from hidden -> moe_intermediate (2048 -> 768).
#: `down` goes from moe_intermediate -> hidden (768 -> 2048).
QWEN3_EXPERT_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("gate", 2048, 768),
    ("up",   2048, 768),
    ("down", 768,  2048),
)


# ---------------------------------------------------------------------------
#  Weight-source helpers
# ---------------------------------------------------------------------------

def load_real_gptq_projection(
    model_path: str,
    *,
    proj_name: str,
    layer_idx: int = 0,
    expert_idx: int = 0,
) -> GPTQLinearWeight:
    loader = ExpertWeightLoader(model_path)
    return loader.load_gptq_expert_linear(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        proj_name=proj_name,
    )


def synthesize_w4a32(
    *,
    in_features: int,
    out_features: int,
    group_size: int = 128,
    seed: int = 0,
) -> GPTQLinearWeight:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.1
    return quantize_symmetric_w4a32(weight, group_size=group_size)


# ---------------------------------------------------------------------------
#  Benchmark cells
# ---------------------------------------------------------------------------

def _cpu_grouped(inputs: torch.Tensor, quantized: GPTQLinearWeight,
                 *, warmup: int, repeats: int) -> dict[str, Any]:
    """CPU baseline — grouped-dequant reference."""
    for _ in range(warmup):
        cpu_w4a32_matvec(inputs, quantized)
    durations: list[float] = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = cpu_w4a32_matvec(inputs, quantized).output
        durations.append(time.perf_counter() - start)
    assert output is not None
    return {
        "output": output,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
    }


def _cpu_dense(inputs: torch.Tensor, quantized: GPTQLinearWeight,
               *, warmup: int, repeats: int) -> dict[str, Any]:
    """Also record CPU dense as a secondary CPU baseline (useful for small shapes)."""
    for _ in range(warmup):
        cpu_w4a32_matvec_dense(inputs, quantized)
    durations: list[float] = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = cpu_w4a32_matvec_dense(inputs, quantized).output
        durations.append(time.perf_counter() - start)
    assert output is not None
    return {
        "output": output,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
    }


def _pim(
    inputs: torch.Tensor,
    quantized: GPTQLinearWeight,
    *,
    rank_count: int,
    profile: str,
    kernel_mode: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Single PIM cell.  Returns ``{"error": str}`` on DPU failure."""
    try:
        runtime = PIMQuantizedRuntime.get_shared(rank_count=rank_count, profile=profile)
    except Exception as exc:  # noqa: BLE001 — we want to record any failure mode
        return {"error": f"runtime_init: {type(exc).__name__}: {exc}"}

    try:
        for _ in range(warmup):
            runtime.linear(inputs, quantized, kernel_mode=kernel_mode)

        durations: list[float] = []
        input_transfer: list[float] = []
        launch: list[float] = []
        output_transfer: list[float] = []
        runtime_total: list[float] = []
        output = None
        for _ in range(repeats):
            start = time.perf_counter()
            output = runtime.linear(inputs, quantized, kernel_mode=kernel_mode)
            durations.append(time.perf_counter() - start)
            phase = runtime.last_profile()
            input_transfer.append(phase["input_transfer_seconds"])
            launch.append(phase["launch_seconds"])
            output_transfer.append(phase["output_transfer_seconds"])
            runtime_total.append(phase["runtime_total_seconds"])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"linear: {type(exc).__name__}: {exc}"}

    assert output is not None
    return {
        "output": output,
        "seconds_avg": sum(durations) / len(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
        "input_transfer_seconds_avg": sum(input_transfer) / len(input_transfer),
        "launch_seconds_avg": sum(launch) / len(launch),
        "output_transfer_seconds_avg": sum(output_transfer) / len(output_transfer),
        "runtime_total_seconds_avg": sum(runtime_total) / len(runtime_total),
        "runtime_dpu_count": runtime.num_dpus(),
    }


# ---------------------------------------------------------------------------
#  Main sweep
# ---------------------------------------------------------------------------

def _make_row(
    *,
    shape_name: str,
    in_features: int,
    out_features: int,
    batch: int,
    rank: int,
    kernel_mode: int,
    cpu_grouped: dict[str, Any],
    cpu_dense: dict[str, Any],
    pim_cell: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shape_name": shape_name,
        "in_features": in_features,
        "out_features": out_features,
        "batch": batch,
        "rank_count": rank,
        "kernel_mode": kernel_mode,
        "cpu_grouped_seconds_avg": cpu_grouped["seconds_avg"],
        "cpu_grouped_seconds_min": cpu_grouped["seconds_min"],
        "cpu_dense_seconds_avg": cpu_dense["seconds_avg"],
    }

    if "error" in pim_cell:
        row["status"] = "pim_error"
        row["pim_error"] = pim_cell["error"]
        return row

    row["status"] = "ok"
    row["pim_seconds_avg"] = pim_cell["seconds_avg"]
    row["pim_seconds_min"] = pim_cell["seconds_min"]
    row["pim_seconds_max"] = pim_cell["seconds_max"]
    row["pim_input_transfer_seconds_avg"] = pim_cell["input_transfer_seconds_avg"]
    row["pim_launch_seconds_avg"] = pim_cell["launch_seconds_avg"]
    row["pim_output_transfer_seconds_avg"] = pim_cell["output_transfer_seconds_avg"]
    row["pim_runtime_total_seconds_avg"] = pim_cell["runtime_total_seconds_avg"]
    row["pim_runtime_dpu_count"] = pim_cell["runtime_dpu_count"]

    # Primary KPI: PIM speedup over CPU grouped baseline (higher is better).
    if pim_cell["seconds_avg"] > 0:
        row["pim_vs_cpu_grouped_ratio"] = cpu_grouped["seconds_avg"] / pim_cell["seconds_avg"]
    else:
        row["pim_vs_cpu_grouped_ratio"] = None

    # Numerical quality: max_abs_error vs CPU grouped.
    try:
        diff = (pim_cell["output"] - cpu_grouped["output"]).abs()
        row["max_abs_error_vs_cpu_grouped"] = float(diff.max().item())
        row["mean_abs_error_vs_cpu_grouped"] = float(diff.mean().item())
    except Exception as exc:  # noqa: BLE001
        row["max_abs_error_vs_cpu_grouped"] = None
        row["mean_abs_error_vs_cpu_grouped"] = None
        row["error_comparison_error"] = f"{type(exc).__name__}: {exc}"

    return row


def sweep(
    *,
    shapes: Sequence[tuple[str, int, int]],
    batches: Sequence[int],
    ranks: Sequence[int],
    kernel_modes: Sequence[int],
    warmup: int,
    repeats: int,
    pim_profile: str,
    model_path: str | None,
    use_synthetic: bool,
    group_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Weight source per shape: load once, reuse across (batch × rank × mode).
    # CPU baseline runs once per (shape × batch).
    weight_cache: dict[str, GPTQLinearWeight] = {}

    def _get_weight(shape_name: str, in_features: int, out_features: int) -> GPTQLinearWeight:
        if shape_name in weight_cache:
            return weight_cache[shape_name]
        if use_synthetic or model_path is None:
            quantized = synthesize_w4a32(
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
                seed=seed + abs(hash(shape_name)) % 1000,
            )
        else:
            quantized = load_real_gptq_projection(model_path, proj_name=shape_name)
            # Sanity-check: real GPTQ may differ from catalog shape across Qwen revisions.
            if (quantized.input_dim, quantized.output_dim) != (in_features, out_features):
                raise RuntimeError(
                    f"GPTQ shape mismatch for {shape_name}: "
                    f"got {(quantized.input_dim, quantized.output_dim)}, "
                    f"expected {(in_features, out_features)}"
                )
        weight_cache[shape_name] = quantized
        return quantized

    generator = torch.Generator().manual_seed(seed)

    for shape_name, in_features, out_features in shapes:
        quantized = _get_weight(shape_name, in_features, out_features)

        for batch in batches:
            inputs = torch.randn(batch, in_features, generator=generator)
            cpu_grouped = _cpu_grouped(inputs, quantized, warmup=warmup, repeats=repeats)
            cpu_dense = _cpu_dense(inputs, quantized, warmup=warmup, repeats=repeats)

            for rank in ranks:
                for kernel_mode in kernel_modes:
                    pim_cell = _pim(
                        inputs,
                        quantized,
                        rank_count=rank,
                        profile=pim_profile,
                        kernel_mode=kernel_mode,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    row = _make_row(
                        shape_name=shape_name,
                        in_features=in_features,
                        out_features=out_features,
                        batch=batch,
                        rank=rank,
                        kernel_mode=kernel_mode,
                        cpu_grouped=cpu_grouped,
                        cpu_dense=cpu_dense,
                        pim_cell=pim_cell,
                    )
                    rows.append(row)
                    # Echo one compact line per row so long sweeps are observable live.
                    ratio = row.get("pim_vs_cpu_grouped_ratio")
                    err = row.get("max_abs_error_vs_cpu_grouped")
                    ratio_s = f"{ratio:.2f}x" if isinstance(ratio, float) else "—"
                    err_s = f"{err:.3e}" if isinstance(err, float) else "—"
                    status = row.get("status", "?")
                    print(
                        f"[sweep] shape={shape_name:<4} in={in_features:<5} out={out_features:<5} "
                        f"batch={batch} rank={rank:<2} mode={kernel_mode} "
                        f"status={status:<9} pim/cpu={ratio_s:<7} max_err={err_s}",
                        flush=True,
                    )

    return rows


# ---------------------------------------------------------------------------
#  Summary helpers (lightweight, full analysis done offline)
# ---------------------------------------------------------------------------

def _summarize_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For each (shape, batch), record the best kernel_mode/rank_count pair."""
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["shape_name"], row["batch"])
        buckets.setdefault(key, []).append(row)

    best: list[dict[str, Any]] = []
    for (shape_name, batch), entries in sorted(buckets.items()):
        valid = [r for r in entries if isinstance(r.get("pim_vs_cpu_grouped_ratio"), float)]
        if not valid:
            continue
        top = max(valid, key=lambda r: r["pim_vs_cpu_grouped_ratio"])
        best.append({
            "shape_name": shape_name,
            "batch": batch,
            "best_kernel_mode": top["kernel_mode"],
            "best_rank_count": top["rank_count"],
            "best_pim_vs_cpu_grouped_ratio": top["pim_vs_cpu_grouped_ratio"],
            "best_max_abs_error_vs_cpu_grouped": top.get("max_abs_error_vs_cpu_grouped"),
        })
    return {"best_by_shape_batch": best}


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0] if __doc__ else "PIM shape sweep",
    )
    parser.add_argument(
        "--model-path",
        default="/home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4",
        help="Local Qwen3 GPTQ checkpoint directory. Used unless --synthetic.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Bypass real checkpoint and use synthetic W4A32 weights. "
             "Only for smoke-running on machines without the real weights.",
    )
    parser.add_argument(
        "--shapes",
        choices=["qwen3"],
        default="qwen3",
        help="Shape catalog. Currently only 'qwen3' is supported.",
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, default=[1, 2, 4, 8],
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=[1, 4, 8, 16, 32],
        help="PIM rank counts to sweep.",
    )
    parser.add_argument(
        "--kernel-modes", nargs="+", type=int, default=[3, 4, 6],
        help="kernel_mode values to sweep. 3=full soft-float, 4=int8 fixed, "
             "6=current 'T-MAC' (likely fake — see ADR-002 §2.2 Gap A).",
    )
    parser.add_argument("--pim-profile", default="", help="libdpu allocation profile.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for synthetic W4A32 weights (ignored for real GPTQ).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out", required=True,
        help="REQUIRED — sweep output path. ADR-002 §5.2 requires every milestone "
             "to archive its sweep under benchmarks/results/<milestone>_<date>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.shapes != "qwen3":  # future: more catalogs
        raise SystemExit(f"Unsupported shape catalog: {args.shapes}")

    if not args.synthetic and not os.path.isdir(args.model_path):
        raise SystemExit(
            f"--model-path does not exist: {args.model_path}\n"
            "Use --synthetic if you are smoke-running on a machine without the checkpoint."
        )

    print(
        f"[sweep] scope: {args.shapes} x batches={args.batches} x ranks={args.ranks} "
        f"x modes={args.kernel_modes} x (warmup={args.warmup}, repeats={args.repeats})",
        flush=True,
    )

    rows = sweep(
        shapes=QWEN3_EXPERT_SHAPES,
        batches=args.batches,
        ranks=args.ranks,
        kernel_modes=args.kernel_modes,
        warmup=args.warmup,
        repeats=args.repeats,
        pim_profile=args.pim_profile,
        model_path=None if args.synthetic else args.model_path,
        use_synthetic=args.synthetic,
        group_size=args.group_size,
        seed=args.seed,
    )

    # Strip tensor outputs before serializing.
    for row in rows:
        row.pop("output", None)

    payload = {
        "torch_version": torch.__version__,
        "model_path": None if args.synthetic else args.model_path,
        "use_synthetic": args.synthetic,
        "pim_profile": args.pim_profile,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "shapes": args.shapes,
        "batches": args.batches,
        "ranks": args.ranks,
        "kernel_modes": args.kernel_modes,
        "results": rows,
        "summary": _summarize_best(rows),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sweep] wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
