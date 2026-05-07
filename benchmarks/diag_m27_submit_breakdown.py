#!/usr/bin/env python3
"""
ADR-002 M-27 Stage 2 diagnostic: break DOWN the cuda_pim submit phase.

Instruments PIMMoEBackend._submit_forward_c_async with sub-phase timers
so we know exactly which part of the 6.5ms-per-step submit is slow:

  sub_0_preamble       : cpu_mask ~~, entry checks
  sub_1_pinned_d2h     : pinned copy + cuda_stream.synchronize
  sub_2_diag_counter   : routed_cpu + sum/any
  sub_3_expert_scan    : 128-expert for loop building activated_cpu_experts
  sub_4_preload        : preload_concat_and_get_slot + preload_and_get_slot
  sub_5_submit_async   : submit_many_fused_silu_async (Python prepare + ctypes)
  sub_6_stash_meta     : _c_async_handle / _c_async_meta assignment

Run in isolation on cuda_pim only.  No cuda.synchronize injected — we
want wall-clock Python time as observed by the main thread.
"""
from __future__ import annotations
import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _install_breakdown_timer(out_path: Path) -> None:
    from nano_ktrans.kernels.pim_moe import PIMMoEBackend
    from nano_ktrans.kernels.pim_quantized_runtime import PIMQuantizedRuntime
    import torch

    stats: dict[str, list[float]] = {}

    def _rec(phase: str, seconds: float) -> None:
        s = stats.setdefault(phase, [0, 0.0])
        s[0] += 1
        s[1] += seconds

    original = PIMMoEBackend._submit_forward_c_async

    def patched(self, hidden_states, topk_ids, topk_weights):
        t_entry = time.perf_counter()
        rt_gate_up = self.quantized_runtime
        rt_down = self.quantized_runtime_down or self.quantized_runtime
        if rt_gate_up is None or rt_down is None:
            return False
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        cpu_mask = ~self.gpu_experts_mask.bool()

        use_pinned_path = (
            hidden_states.is_cuda
            and torch.cuda.is_available()
            and self.enable_m25_pinned_d2h
        )
        _rec("sub_0_preamble", time.perf_counter() - t_entry)

        # sub_1_pinned_d2h
        t = time.perf_counter()
        if use_pinned_path:
            flat_cpu, topk_ids_cpu, topk_weights_cpu = self._acquire_pinned_submit_buffers(
                batch_size=batch_size,
                hidden_size=flat.shape[-1],
                top_k=topk_ids.shape[-1],
            )
            flat_cpu.copy_(flat.to(dtype=torch.float32), non_blocking=True)
            topk_ids_cpu.copy_(topk_ids.to(dtype=torch.long), non_blocking=True)
            topk_weights_cpu.copy_(topk_weights.to(dtype=torch.float32), non_blocking=True)
            torch.cuda.current_stream().synchronize()
        else:
            flat_cpu = flat.to("cpu", dtype=torch.float32)
            topk_ids_cpu = topk_ids.to("cpu", dtype=torch.long)
            topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)
        _rec("sub_1_pinned_d2h", time.perf_counter() - t)

        # sub_2_diag_counter
        t = time.perf_counter()
        routed_cpu = cpu_mask[topk_ids_cpu]
        self.offloaded_pairs += int(routed_cpu.sum().item())
        self.offloaded_tokens += int(routed_cpu.any(dim=1).sum().item())
        _rec("sub_2_diag_counter", time.perf_counter() - t)

        # sub_3_expert_scan
        t = time.perf_counter()
        # M-27 Stage B: vectorised — iterate only routed-this-step
        # unique ids (small set) instead of all num_experts.
        activated_cpu_experts = []
        unique_ids = torch.unique(topk_ids_cpu).tolist()
        for expert_idx in unique_ids:
            if not cpu_mask[expert_idx]:
                continue
            cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
            if cpu_slot is None:
                continue
            match = topk_ids_cpu == expert_idx
            token_indices = torch.where(match.any(dim=1))[0]
            if len(token_indices) == 0:
                continue
            if cpu_slot not in self._gptq_experts:
                _rec("sub_3_expert_scan", time.perf_counter() - t)
                return False
            activated_cpu_experts.append((expert_idx, cpu_slot, token_indices, match))
        _rec("sub_3_expert_scan", time.perf_counter() - t)

        if not activated_cpu_experts:
            self._fallback_output = torch.zeros(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            return True

        # sub_4_preload
        t = time.perf_counter()
        kernel_mode = 4
        requests = []
        per_expert_records = []
        pre_hits_gu = rt_gate_up.preload_hits
        pre_miss_gu = rt_gate_up.preload_misses
        pre_profile_gu = self._snapshot_quantized_profile(rt_gate_up)
        pre_hits_dn = rt_down.preload_hits
        pre_miss_dn = rt_down.preload_misses
        pre_profile_dn = self._snapshot_quantized_profile(rt_down)
        for expert_idx, cpu_slot, token_indices, match in activated_cpu_experts:
            gptq = self._gptq_experts.get(cpu_slot)
            if gptq is None:
                _rec("sub_4_preload", time.perf_counter() - t)
                return False
            states = flat_cpu[token_indices]
            base_eid = self._expert_id(cpu_slot)
            gate_up_eid = base_eid ^ 0x1212121212121212
            down_eid = base_eid ^ 0x3333333333333333
            gu_slot, gu_padded_in, gu_concat, gate_cols, up_cols = \
                rt_gate_up.preload_concat_and_get_slot(
                    gate_up_eid, gptq["gate"], gptq["up"], kernel_mode=kernel_mode,
                )
            dn_slot, dn_padded_in, dn_padded_out, dn_orig_out = \
                rt_down.preload_and_get_slot(down_eid, gptq["down"], kernel_mode)
            if gate_cols != up_cols or dn_padded_in != up_cols:
                _rec("sub_4_preload", time.perf_counter() - t)
                return False
            requests.append(
                (states, gu_slot, gu_padded_in, gu_concat,
                 gate_cols, up_cols, dn_slot, dn_padded_in, dn_padded_out)
            )
            per_expert_records.append({
                "token_indices": token_indices,
                "match": match,
                "dn_orig_out": dn_orig_out,
            })
        _rec("sub_4_preload", time.perf_counter() - t)

        # sub_5_submit_async
        t = time.perf_counter()
        handle = PIMQuantizedRuntime.submit_many_fused_silu_async(
            rt_gate_up, rt_down, requests,
        )
        _rec("sub_5_submit_async", time.perf_counter() - t)

        # sub_6_stash_meta
        t = time.perf_counter()
        self._c_async_handle = handle
        self._c_async_meta = {
            "per_expert_records": per_expert_records,
            "topk_weights_cpu": topk_weights_cpu,
            "batch_size": batch_size,
            "hidden_size": self.hidden_size,
            "device": hidden_states.device,
            "hidden_dtype": hidden_states.dtype,
            "hidden_shape": hidden_states.shape,
            "rt_gate_up": rt_gate_up,
            "rt_down": rt_down,
            "pre_hits_gu": pre_hits_gu,
            "pre_miss_gu": pre_miss_gu,
            "pre_profile_gu": pre_profile_gu,
            "pre_hits_dn": pre_hits_dn,
            "pre_miss_dn": pre_miss_dn,
            "pre_profile_dn": pre_profile_dn,
            "n": len(per_expert_records),
        }
        self.c_async_submit_count += 1
        _rec("sub_6_stash_meta", time.perf_counter() - t)
        return True

    PIMMoEBackend._submit_forward_c_async = patched

    def _dump():
        payload = {
            p: {"count": c, "total_s": t, "mean_ms": (t / c * 1000) if c else None}
            for p, (c, t) in stats.items()
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        sys.stderr.write(f"[diag_m27_sub] wrote {out_path}\n")

    atexit.register(_dump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--routing-freq-json", default="benchmarks/results/routing_freq_qwen3_30b_m23_mean.json")
    ap.add_argument("--out", default="benchmarks/results/diag_m27_submit_breakdown.json")
    ap.add_argument("--bench-out", default="benchmarks/results/diag_m27_submit_breakdown_bench.json")
    args = ap.parse_args()

    env = os.environ.copy()
    env["BENCH_DIAG_SUB_STATS"] = str(Path(args.out).resolve())
    env["PYTHONUNBUFFERED"] = "1"

    shim = f"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(r'{REPO}') / 'benchmarks'))
from diag_m27_submit_breakdown import _install_breakdown_timer
_install_breakdown_timer(Path(os.environ['BENCH_DIAG_SUB_STATS']))
import runpy
sys.argv = [
    'benchmark_inference',
    '--model-path', r'{args.model_path}',
    '--backends', 'cuda_pim',
    '--offload-device-experts', '92',
    '--routing-freq-json', r'{args.routing_freq_json}',
    '--pim-rank-count', '1',
    '--pim-layer-group-size', '48',
    '--max-new-tokens', '{args.max_new_tokens}',
    '--warmup', '0',
    '--repeats', '1',
    '--pim-enable-c-async',
    '--pim-enable-m25-pinned',
    '--json-out', r'{args.bench_out}',
]
runpy.run_path(r'{REPO}/benchmarks/benchmark_inference.py', run_name='__main__')
"""
    print("launching cuda_pim with sub-phase breakdown...")
    t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, "-u", "-c", shim], env=env, cwd=str(REPO), timeout=1800)
    print(f"done in {time.perf_counter() - t0:.1f}s rc={rc.returncode}")
    stats = json.loads(Path(args.out).read_text())
    print("\n=== submit sub-phase breakdown ===")
    total = sum(v["total_s"] for v in stats.values())
    print(f"{'phase':<24}{'total_s':>10}{'count':>8}{'mean_ms':>12}{'pct':>8}")
    for p in ["sub_0_preamble", "sub_1_pinned_d2h", "sub_2_diag_counter",
              "sub_3_expert_scan", "sub_4_preload", "sub_5_submit_async",
              "sub_6_stash_meta"]:
        v = stats.get(p, {"total_s": 0, "count": 0, "mean_ms": 0})
        tot = v.get("total_s") or 0
        pct = (tot / total * 100) if total else 0
        print(f"{p:<24}{tot:>10.3f}{v.get('count', 0):>8}{v.get('mean_ms', 0) or 0:>12.4f}{pct:>7.1f}%")
    print(f"{'TOTAL':<24}{total:>10.3f}")


if __name__ == "__main__":
    main()
