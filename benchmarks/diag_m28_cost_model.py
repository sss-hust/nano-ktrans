#!/usr/bin/env python3
"""
ADR-002 M-28 cost-model microbench.

Goal: measure the *primitive* costs (per-expert) needed for a Fiddler-style
runtime decision in HybridMoE.forward, without loading the full Qwen3-30B
model. We synthesise GPTQ-shaped weights matching the real model dims
(hidden=2048, intermediate=768, INT4 group_size=128) and time:

    a) t_h2d_act    - H2D copy of one decode-step activation [1, 2048] fp16->cuda
    b) t_d2h_act    - D2H copy of one expert output [1, 2048] cuda->cpu fp32
    c) t_gpu_expert - one expert (gate+up+silu*up+down) on GPU in fp16/bf16
    d) t_cpu_expert - one GPTQ expert via cpu_w4a32_matvec (3 mat-vecs)
    e) t_pim_pre    - PIMQuantizedRuntime.preload (cold; gate_up + down)
    f) t_pim_inf    - PIMQuantizedRuntime.infer (one expert, kernel exec only)

Outputs benchmarks/results/diag_m28_cost_model.json plus a printed table
that gets pasted into ADR-002 §M-28.

Usage:
    python3 benchmarks/diag_m28_cost_model.py [--repeat N] [--no-pim] [--no-gpu]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent

# ── Model dimensions (Qwen3-30B-A3B-GPTQ) ─────────────────────────────
HIDDEN = 2048
INTER = 768
GROUP_SIZE = 128
BITS = 4

# ── synthetic GPTQ weight builder ─────────────────────────────────────
from nano_ktrans.kernels.weight_loader import GPTQLinearWeight  # noqa: E402


def make_gptq(out_dim: int, in_dim: int, prefix: str) -> GPTQLinearWeight:
    """Build a GPTQLinearWeight with the same dtype/shape conventions
    used by the real loader. Values are random — we only need it for
    timing, not numerical comparison."""
    values_per_word = 32 // BITS  # 8
    in_words = in_dim // values_per_word  # 256 for hidden=2048
    qweight = torch.randint(
        0, 2**31 - 1, (out_dim, in_words), dtype=torch.int32
    )
    num_groups = in_dim // GROUP_SIZE  # 16 for hidden, 6 for inter
    scales = (torch.rand(out_dim, num_groups, dtype=torch.float32) * 0.01) + 1e-4
    return GPTQLinearWeight(
        qweight=qweight,
        scales=scales,
        zero_points=None,
        group_size=GROUP_SIZE,
        bits=BITS,
        sym=True,
        linear_prefix=prefix,
    )


# ── (a)+(b) GPU copy primitives ───────────────────────────────────────
def bench_copies(repeat: int, device: torch.device) -> dict:
    if device.type != "cuda":
        return {"skipped": "no cuda device"}
    # decode batch = 1 token
    x_gpu = torch.randn(1, HIDDEN, dtype=torch.float16, device=device)
    x_cpu = torch.empty(1, HIDDEN, dtype=torch.float32, pin_memory=True)
    out_gpu = torch.randn(1, HIDDEN, dtype=torch.float16, device=device)
    out_cpu = torch.empty(1, HIDDEN, dtype=torch.float32, pin_memory=True)

    # warmup
    for _ in range(20):
        x_cpu.copy_(x_gpu.to(dtype=torch.float32), non_blocking=True)
        out_cpu.copy_(out_gpu.to(dtype=torch.float32), non_blocking=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        x_cpu.copy_(x_gpu.to(dtype=torch.float32), non_blocking=True)
    torch.cuda.synchronize()
    t_d2h = (time.perf_counter() - t0) / repeat * 1000

    t0 = time.perf_counter()
    for _ in range(repeat):
        x_gpu.copy_(x_cpu.to(dtype=torch.float16), non_blocking=True)
    torch.cuda.synchronize()
    t_h2d = (time.perf_counter() - t0) / repeat * 1000

    return {
        "t_d2h_act_ms": round(t_d2h, 4),
        "t_h2d_act_ms": round(t_h2d, 4),
        "bytes_per_copy": 1 * HIDDEN * 4,  # fp32
    }


# ── (c) GPU expert ─────────────────────────────────────────────────────
def bench_gpu_expert(repeat: int, device: torch.device) -> dict:
    if device.type != "cuda":
        return {"skipped": "no cuda device"}
    dtype = torch.float16
    w1 = torch.randn(INTER, HIDDEN, dtype=dtype, device=device) * 0.02
    w3 = torch.randn(INTER, HIDDEN, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(HIDDEN, INTER, dtype=dtype, device=device) * 0.02
    x = torch.randn(1, HIDDEN, dtype=dtype, device=device)

    def expert_fwd(h):
        gate = torch.nn.functional.silu(torch.nn.functional.linear(h, w1))
        up = torch.nn.functional.linear(h, w3)
        return torch.nn.functional.linear(gate * up, w2)

    for _ in range(20):
        _ = expert_fwd(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = expert_fwd(x)
    torch.cuda.synchronize()
    return {"t_gpu_expert_ms": round((time.perf_counter() - t0) / repeat * 1000, 4)}


# ── (d) CPU GPTQ expert ────────────────────────────────────────────────
def bench_cpu_expert(repeat: int) -> dict:
    from nano_ktrans.kernels.quantized_ops import cpu_w4a32_matvec

    gate = make_gptq(INTER, HIDDEN, "gate")
    up = make_gptq(INTER, HIDDEN, "up")
    down = make_gptq(HIDDEN, INTER, "down")
    x = torch.randn(1, HIDDEN, dtype=torch.float32)

    def fwd(x):
        g = cpu_w4a32_matvec(x, gate).output
        u = cpu_w4a32_matvec(x, up).output
        h = torch.nn.functional.silu(g) * u
        return cpu_w4a32_matvec(h, down).output

    for _ in range(3):
        _ = fwd(x)

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fwd(x)
    return {"t_cpu_expert_ms": round((time.perf_counter() - t0) / repeat * 1000, 4)}


# ── (e)+(f) PIM preload + infer ───────────────────────────────────────
def bench_pim(repeat: int) -> dict:
    try:
        from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"import failed: {exc}"}

    try:
        rt_gu = PIMQuantizedRuntime(rank_count=4, instance_key="m28-gu")
        rt_dn = PIMQuantizedRuntime(rank_count=4, instance_key="m28-dn")
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"runtime alloc failed: {exc}"}

    # Build a small pool of distinct experts so we exercise miss-path
    n_experts = 4
    gate_up_packed = []
    down_packed = []
    for eid in range(1, n_experts + 1):
        # gate and up concatenated -> shape (2*INTER, HIDDEN)
        gu = make_gptq(2 * INTER, HIDDEN, f"L0.E{eid}.gu")
        dn = make_gptq(HIDDEN, INTER, f"L0.E{eid}.dn")
        gate_up_packed.append((eid, gu))
        down_packed.append((eid, dn))

    # warm preload once each (so the host-side _weight_cache is filled
    # and we measure pure DMA on subsequent calls — this is what the
    # real hot path does after first encounter).
    for eid, gu in gate_up_packed:
        try:
            rt_gu.preload(eid, gu, kernel_mode=4)
        except Exception as exc:
            return {"skipped": f"preload failed: {exc}"}
    for eid, dn in down_packed:
        try:
            rt_dn.preload(eid, dn, kernel_mode=4)
        except Exception:
            pass

    # Force evictions to measure miss-path: cycle through MRAM slots
    # large enough that the next preload is guaranteed cold for the
    # measured eid. With NUM_SLOTS=128 and 4 experts they all hit, so
    # we explicitly bench the *first* (cold) preload by recreating
    # runtime — too heavy. Instead measure the warm path which the
    # cost model needs (real submit_forward sees mostly hits at
    # M-27 Stage C).
    measurements = {}

    # warm preload (hit) cost — what M-27 Stage C achieves 45% of the time
    eid_hit = gate_up_packed[0][0]
    gu_hit = gate_up_packed[0][1]
    dn_hit = down_packed[0][1]
    for _ in range(3):
        rt_gu.preload(eid_hit, gu_hit, kernel_mode=4)
        rt_dn.preload(eid_hit, dn_hit, kernel_mode=4)
    t0 = time.perf_counter()
    for _ in range(repeat):
        rt_gu.preload(eid_hit, gu_hit, kernel_mode=4)
        rt_dn.preload(eid_hit, dn_hit, kernel_mode=4)
    measurements["t_pim_preload_warm_hit_ms"] = round(
        (time.perf_counter() - t0) / repeat * 1000, 4
    )

    # PIM infer (1 token, gate_up + down via fused if available)
    try:
        x = torch.randn(1, HIDDEN, dtype=torch.float32)
        # Try the lighter primitive: infer() on already-resident weights.
        # We don't have direct access to a fused-silu wrapper here, but
        # measuring two infers separately gives a tight upper bound on
        # the per-expert PIM kernel cost.
        _ = rt_gu.infer(x)
        h = torch.randn(1, INTER, dtype=torch.float32)
        _ = rt_dn.infer(h)
        for _ in range(3):
            _ = rt_gu.infer(x)
            _ = rt_dn.infer(h)
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = rt_gu.infer(x)
            _ = rt_dn.infer(h)
        measurements["t_pim_infer_per_expert_ms"] = round(
            (time.perf_counter() - t0) / repeat * 1000, 4
        )
    except Exception as exc:  # noqa: BLE001
        measurements["t_pim_infer_per_expert_ms_skipped"] = str(exc)

    return measurements


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--no-pim", action="store_true")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--out", default=str(REPO / "benchmarks/results/diag_m28_cost_model.json"))
    args = p.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")
    print(f"[m28] device={device}, repeat={args.repeat}")

    out = {
        "model_dims": {
            "hidden": HIDDEN,
            "intermediate": INTER,
            "group_size": GROUP_SIZE,
            "bits": BITS,
        },
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    print("\n[m28] (a)+(b) H2D / D2H copy ...")
    out["copies"] = bench_copies(args.repeat, device)
    print(f"  -> {out['copies']}")

    if not args.no_gpu:
        print("\n[m28] (c) GPU expert (fp16) ...")
        out["gpu_expert"] = bench_gpu_expert(args.repeat, device)
        print(f"  -> {out['gpu_expert']}")

    print("\n[m28] (d) CPU GPTQ expert (W4A32) ...")
    out["cpu_expert"] = bench_cpu_expert(min(args.repeat, 50))
    print(f"  -> {out['cpu_expert']}")

    if not args.no_pim:
        print("\n[m28] (e)+(f) PIM preload + infer ...")
        out["pim"] = bench_pim(min(args.repeat, 100))
        print(f"  -> {out['pim']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[m28] wrote {out_path}")


if __name__ == "__main__":
    main()
