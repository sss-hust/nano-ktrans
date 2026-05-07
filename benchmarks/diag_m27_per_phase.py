#!/usr/bin/env python3
"""
ADR-002 M-27 diagnostic: fine-grained per-phase timer of HybridMoE.forward,
parameterised by backend.  Run as a child of benchmark_inference.py so
the LLM init path matches production exactly.

Strategy: monkey-patch HybridMoE.forward to bracket its internal phases
before run_benchmark() is called, via a BENCH_DIAG_PHASES=1 env var.

Each backend is run in its own process to avoid CUDA memory accumulation.

Output: per-phase total seconds + sample mean for cuda_pim (with M-25)
and cuda_cpu_offload, directly comparable.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _install_phase_timer(out_path: Path) -> None:
    """Monkey-patch HybridMoE.forward with per-phase wall-clock timers.

    Must be called BEFORE any HybridMoE instance is built.  Accumulates
    counts + total seconds per phase, writes to ``out_path`` at exit.
    """
    from nano_ktrans.layers.hybrid_moe import HybridMoE
    from nano_ktrans.utils.context import get_context

    phase_stats: dict[str, list[float]] = {}

    def _record(phase: str, seconds: float) -> None:
        slot = phase_stats.setdefault(phase, [0, 0.0])
        slot[0] += 1
        slot[1] += seconds

    original_forward = HybridMoE.forward  # noqa: F841 (kept for reference)

    def patched_forward(self, hidden_states, router_logits):
        use_cuda = hidden_states.is_cuda and torch.cuda.is_available()

        def _sync_now() -> None:
            if use_cuda:
                torch.cuda.synchronize()

        # -------- Step 1: routing + scheduler bookkeeping --------
        _sync_now()
        t0 = time.perf_counter()
        if self.router_use_softmax:
            router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
            topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_weights, dim=-1)

        context = get_context()
        phase_name = "prefill" if context.is_prefill else "decode"
        active_experts = {int(expert_idx) for expert_idx in torch.unique(topk_ids).tolist()}
        self._record_router_probs(router_logits)
        with self._pipeline_lock:
            self._apply_queued_migrations(hidden_states, active_experts, phase=phase_name)
            if self.dynamic_expert_scheduler is not None and self.dynamic_expert_scheduler.enabled:
                self.dynamic_expert_scheduler.observe(
                    self.layer_idx,
                    topk_ids,
                    phase=phase_name,
                    topk_weights=topk_weights,
                )
                self._request_prefetch_candidates(phase=phase_name)
                planned_ops = self.dynamic_expert_scheduler.plan_layer(self.layer_idx, phase=phase_name)
                if planned_ops and self.offload_backend is not None:
                    for op in planned_ops:
                        from nano_ktrans.scheduler.expert_residency import ExpertResidency
                        if (
                            op.dst == ExpertResidency.GPU
                            and not self.materialization_manager.has_cached(self.layer_idx, int(op.expert_idx))
                        ):
                            self._request_prefetch(op.expert_idx)
                    self.offload_backend.queue_migration_plan(planned_ops, phase=phase_name)
            self._request_map_store_prefetch()
            gpu_experts_mask_snapshot = self.gpu_experts_mask.detach().clone()
            gpu_experts_snapshot = dict(self.gpu_experts.items())

        _sync_now()
        if not context.is_prefill:
            _record("step_1_routing", time.perf_counter() - t0)

        # -------- Step 2: offload submit --------
        _sync_now()
        t1 = time.perf_counter()
        cuda_stream = None
        if self.offload_backend is not None:
            if use_cuda:
                cuda_stream = torch.cuda.current_stream().cuda_stream
            self.offload_backend.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        if not context.is_prefill:
            # No cuda_sync here — this is the MAIN-THREAD wall-clock of submit.
            _record("step_2_submit", time.perf_counter() - t1)

        # -------- Step 3: GPU expert loop --------
        t2 = time.perf_counter()
        final_gpu_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            topk_ids, num_classes=self.num_experts
        ).sum(dim=1).bool()

        for expert_idx in range(self.num_experts):
            if not gpu_experts_mask_snapshot[expert_idx]:
                continue
            expert_key = str(expert_idx)
            if expert_key not in gpu_experts_snapshot:
                continue
            token_indices = torch.where(expert_mask[:, expert_idx])[0]
            if len(token_indices) == 0:
                continue
            current_state = hidden_states[token_indices]
            expert_output = gpu_experts_snapshot[expert_key](current_state)
            expert_match = (topk_ids[token_indices] == expert_idx)
            weights = topk_weights[token_indices][expert_match]
            expert_output = expert_output * weights.unsqueeze(1)
            final_gpu_states.index_add_(0, token_indices, expert_output)

        _sync_now()  # capture GPU work completion
        if not context.is_prefill:
            _record("step_3_gpu_expert_loop", time.perf_counter() - t2)

        # -------- Step 4: sync --------
        t3 = time.perf_counter()
        if self.offload_backend is not None:
            cpu_output = self.offload_backend.sync_forward(hidden_states, cuda_stream)
        else:
            cpu_output = torch.zeros_like(hidden_states)
        _sync_now()
        if not context.is_prefill:
            _record("step_4_sync", time.perf_counter() - t3)

        # -------- Step 5: merge --------
        t4 = time.perf_counter()
        result = final_gpu_states + cpu_output
        _sync_now()
        if not context.is_prefill:
            _record("step_5_merge", time.perf_counter() - t4)

        return result

    HybridMoE.forward = patched_forward

    # Register atexit to dump stats.
    import atexit

    def _dump() -> None:
        payload = {
            phase: {"count": cnt, "total_s": tot, "mean_ms": (tot / cnt * 1000) if cnt else None}
            for phase, (cnt, tot) in phase_stats.items()
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        sys.stderr.write(f"[diag_m27] wrote phase stats to {out_path}\n")

    atexit.register(_dump)


def _run_backend_child(
    *,
    backend_tag: str,
    backend: str,
    model_path: str,
    routing_freq_json: str,
    extra_args: list[str],
    max_new_tokens: int,
    stats_path: Path,
    bench_json_path: Path,
) -> dict[str, Any]:
    """Launch benchmark_inference in a child process with phase timer installed."""
    env = os.environ.copy()
    env["BENCH_DIAG_PHASES_STATS"] = str(stats_path)
    env["PYTHONUNBUFFERED"] = "1"
    # Use an injection shim so phase timer installs before HybridMoE appears.
    shim = f"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(r'{REPO}') / 'benchmarks'))
from diag_m27_per_phase import _install_phase_timer
_install_phase_timer(Path(os.environ['BENCH_DIAG_PHASES_STATS']))
# Fall through to benchmark_inference main
import runpy
sys.argv = [
    'benchmark_inference',
    '--model-path', r'{model_path}',
    '--backends', r'{backend}',
    '--offload-device-experts', '92',
    '--routing-freq-json', r'{routing_freq_json}',
    '--pim-rank-count', '1',
    '--pim-layer-group-size', '48',
    '--max-new-tokens', '{max_new_tokens}',
    '--warmup', '0',
    '--repeats', '1',
    '--json-out', r'{bench_json_path}',
] + {extra_args!r}
runpy.run_path(r'{REPO}/benchmarks/benchmark_inference.py', run_name='__main__')
"""
    sys.stderr.write(f"[diag_m27] launching {backend_tag}...\n")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", "-c", shim],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(f"[diag_m27] {backend_tag} FAILED rc={proc.returncode}\n")
        sys.stderr.write("stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-30:]) + "\n")
        sys.stderr.write("stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]) + "\n")
        return {"tag": backend_tag, "error": proc.returncode, "elapsed": elapsed}

    # Parse benchmark json for tps + decode_s
    try:
        bench = json.loads(Path(bench_json_path).read_text())
        run0 = bench["results"][0]["runs"][0]
        decode_s = run0.get("decode_seconds", None)
        tps = run0.get("decode_tokens_per_second", None)
        prefill_s = run0.get("prefill_seconds", None)
    except Exception as e:
        decode_s = tps = prefill_s = None
        sys.stderr.write(f"[diag_m27] {backend_tag} couldn't parse bench json: {e}\n")

    try:
        stats = json.loads(Path(stats_path).read_text())
    except Exception:
        stats = {}

    sys.stderr.write(f"[diag_m27] {backend_tag} done in {elapsed:.1f}s, tps={tps}\n")
    return {
        "tag": backend_tag,
        "elapsed": elapsed,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tps": tps,
        "phase_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--routing-freq-json",
        default="benchmarks/results/routing_freq_qwen3_30b_m23_mean.json",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/results/diag_m27_per_phase.json",
    )
    args = parser.parse_args()

    results = []

    # cuda_pim with M-24 A + M-25
    results.append(_run_backend_child(
        backend_tag="cuda_pim_m25",
        backend="cuda_pim",
        model_path=args.model_path,
        routing_freq_json=args.routing_freq_json,
        extra_args=[
            "--pim-enable-c-async",
            "--pim-enable-m25-pinned",
        ],
        max_new_tokens=args.max_new_tokens,
        stats_path=Path("benchmarks/results/diag_m27_phase_cuda_pim_m25.json"),
        bench_json_path=Path("benchmarks/results/diag_m27_bench_cuda_pim_m25.json"),
    ))

    # cuda_cpu_offload
    results.append(_run_backend_child(
        backend_tag="cuda_cpu_offload",
        backend="cuda_cpu_offload",
        model_path=args.model_path,
        routing_freq_json=args.routing_freq_json,
        extra_args=[],
        max_new_tokens=args.max_new_tokens,
        stats_path=Path("benchmarks/results/diag_m27_phase_cuda_cpu_offload.json"),
        bench_json_path=Path("benchmarks/results/diag_m27_bench_cuda_cpu_offload.json"),
    ))

    # Print comparison
    print("\n===== M-27 per-phase diagnostic =====")
    phases = [
        "step_1_routing",
        "step_2_submit",
        "step_3_gpu_expert_loop",
        "step_4_sync",
        "step_5_merge",
    ]

    def _s(r, p, k="total_s"):
        return ((r.get("phase_stats") or {}).get(p) or {}).get(k, 0.0) or 0.0

    print(f"{'phase':<28}{'cuda_pim (M-25)':>22}{'cuda_cpu_offload':>22}{'delta (s)':>15}{'delta (ms/step)':>18}")
    for phase in phases:
        p = _s(results[0], phase)
        c = _s(results[1], phase)
        p_cnt = ((results[0].get("phase_stats") or {}).get(phase) or {}).get("count", 0) or 0
        c_cnt = ((results[1].get("phase_stats") or {}).get(phase) or {}).get("count", 0) or 0
        p_mean = p / p_cnt * 1000 if p_cnt else 0
        c_mean = c / c_cnt * 1000 if c_cnt else 0
        print(f"{phase:<28}{p:>16.3f}s ({p_cnt:>3}){c:>16.3f}s ({c_cnt:>3}){p - c:>+15.3f}{p_mean - c_mean:>+18.3f}")
    sub_p = sum(_s(results[0], p) for p in phases)
    sub_c = sum(_s(results[1], p) for p in phases)
    print(f"{'sum of phases':<28}{sub_p:>22.3f}{sub_c:>22.3f}{sub_p - sub_c:>+15.3f}")
    d_p = results[0].get("decode_s") or 0.0
    d_c = results[1].get("decode_s") or 0.0
    print(f"{'total decode_s (bench)':<28}{d_p:>22.3f}{d_c:>22.3f}{d_p - d_c:>+15.3f}")
    print(f"{'decode_tps':<28}{results[0].get('tps', 0) or 0:>22.4f}{results[1].get('tps', 0) or 0:>22.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
