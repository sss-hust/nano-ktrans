#!/usr/bin/env python3
"""
ADR-002 M-11 — residency sweep driver.

This script automates the low-risk M-11 track discovered in M-10:
`offload_device_experts=32` beat the previous M-4 peak without any
kernel changes.  We need a reproducible sweep over GPU-resident expert
counts and prompt lengths before promoting a new default.

The script shells out to `benchmark_inference.py` once per cell instead
of importing it in-process.  Reason: each cell may allocate DPU ranks,
CUDA memory, and large model state; subprocess boundaries give us clean
teardown and let OOM/error cases be captured as data instead of killing
the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_LIBRARY: dict[str, str] = {
    "short": "请解释一下如何写出结构清晰的 Python 脚本。",
    "medium": (
        "请用中文详细解释 MoE 大模型推理中专家路由、显存占用、CPU offload "
        "和 PIM 加速之间的关系，并给出一个简短例子。"
    ),
    "long": (
        "请写一篇技术说明，面向系统工程师介绍大语言模型推理优化。"
        "内容需要覆盖模型量化、KV cache、批处理、MoE 专家路由、GPU/CPU "
        "协同、内存带宽瓶颈、以及如何设计基准测试。请逐条展开。" * 6
    ),
}


def _run_cell(
    *,
    repo_root: Path,
    model_path: str,
    prompt_name: str,
    prompt: str,
    offload_device_experts: int,
    max_new_tokens: int,
    timeout_seconds: int,
    cell_json_path: Path,
    pim_layer_group_size: int,
    pim_async: bool,
    speculative_preload: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "benchmarks" / "benchmark_inference.py"),
        "--model-path", model_path,
        "--backends", "cuda_pim",
        "--offload-device-experts", str(offload_device_experts),
        "--max-new-tokens", str(max_new_tokens),
        "--prompt", prompt,
        "--pim-layer-group-size", str(pim_layer_group_size),
        "--json-out", str(cell_json_path),
    ]
    if pim_async:
        cmd.append("--pim-enable-async-submit")
    else:
        cmd.append("--no-pim-async-submit")
    if speculative_preload:
        cmd.append("--pim-enable-speculative-preload-gptq")
    else:
        cmd.append("--no-pim-speculative-preload-gptq")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        return {
            "prompt_name": prompt_name,
            "offload_device_experts": offload_device_experts,
            "status": "timeout",
            "elapsed_seconds": time.perf_counter() - started,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "cell_json_path": str(cell_json_path),
        }

    row: dict[str, Any] = {
        "prompt_name": prompt_name,
        "offload_device_experts": offload_device_experts,
        "process_exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "cell_json_path": str(cell_json_path),
    }

    if not cell_json_path.exists():
        row.update({"status": "missing_json"})
        return row

    try:
        payload = json.loads(cell_json_path.read_text())
        result = payload["results"][0]
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "invalid_json", "error": f"{type(exc).__name__}: {exc}"})
        return row

    row["status"] = result.get("status", "unknown")
    row["backend_status"] = result.get("status")
    row["error_type"] = result.get("error_type")
    row["error"] = result.get("error")

    runs = result.get("runs") or []
    if runs:
        run0 = runs[0]
        row.update({
            "prompt_tokens": run0.get("prompt_tokens"),
            "generated_tokens": run0.get("generated_tokens"),
            "prefill_seconds": run0.get("prefill_seconds"),
            "decode_seconds": run0.get("decode_seconds"),
            "decode_tokens_per_second": run0.get("decode_tokens_per_second"),
            "cuda_max_memory_bytes": run0.get("cuda_max_memory_bytes"),
        })

    layers = (result.get("offload_diagnostics") or {}).get("layers", [])
    row["layer_count"] = len(layers)
    row["real_dpu_quantized_calls"] = sum(
        (layer.get("backend") or {}).get("real_dpu_quantized_calls", 0) or 0
        for layer in layers
    )
    row["quantized_preload_hits_local"] = sum(
        (layer.get("backend") or {}).get("quantized_preload_hits_local", 0) or 0
        for layer in layers
    )
    row["quantized_preload_misses_local"] = sum(
        (layer.get("backend") or {}).get("quantized_preload_misses_local", 0) or 0
        for layer in layers
    )
    row["locality_decode_jaccard_count"] = sum(
        (layer.get("backend") or {}).get("locality_decode_jaccard_count", 0) or 0
        for layer in layers
    )
    row["locality_decode_jaccard_sum"] = sum(
        (layer.get("backend") or {}).get("locality_decode_jaccard_sum", 0.0) or 0.0
        for layer in layers
    )
    if row["locality_decode_jaccard_count"]:
        row["locality_decode_jaccard_mean"] = (
            row["locality_decode_jaccard_sum"] / row["locality_decode_jaccard_count"]
        )
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("decode_tokens_per_second"), (int, float))]
    best = max(ok_rows, key=lambda r: r["decode_tokens_per_second"], default=None)
    by_prompt: dict[str, dict[str, Any]] = {}
    for prompt_name in sorted({str(r.get("prompt_name")) for r in rows}):
        prs = [r for r in ok_rows if r.get("prompt_name") == prompt_name]
        by_prompt[prompt_name] = {
            "ok_count": len(prs),
            "best": max(prs, key=lambda r: r["decode_tokens_per_second"], default=None),
            "oom_count": sum(1 for r in rows if r.get("prompt_name") == prompt_name and r.get("backend_status") == "oom"),
            "error_count": sum(1 for r in rows if r.get("prompt_name") == prompt_name and r.get("status") not in ("ok", "oom")),
        }
    return {
        "cell_count": len(rows),
        "ok_count": len(ok_rows),
        "oom_count": sum(1 for r in rows if r.get("backend_status") == "oom"),
        "error_count": sum(1 for r in rows if r.get("status") not in ("ok", "oom")),
        "best": best,
        "by_prompt": by_prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADR-002 M-11 residency sweep")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--offload-values", nargs="+", type=int, default=[2, 16, 32, 48, 64])
    parser.add_argument("--prompt-profiles", nargs="+", choices=sorted(PROMPT_LIBRARY), default=["short"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--pim-layer-group-size", type=int, default=48)
    parser.add_argument("--pim-async", action="store_true", help="Enable Python-level async PIM submit (default false).")
    parser.add_argument("--speculative-preload", action="store_true", help="Enable GPTQ speculative preload (default false).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_dir = out_path.with_suffix("")
    cell_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for prompt_name in args.prompt_profiles:
        prompt = PROMPT_LIBRARY[prompt_name]
        for offload in args.offload_values:
            cell_json = cell_dir / f"{prompt_name}_offload{offload}.json"
            print(f"[sweep] prompt={prompt_name:<6} offload={offload:<4} -> {cell_json}", flush=True)
            row = _run_cell(
                repo_root=repo_root,
                model_path=args.model_path,
                prompt_name=prompt_name,
                prompt=prompt,
                offload_device_experts=offload,
                max_new_tokens=args.max_new_tokens,
                timeout_seconds=args.timeout_seconds,
                cell_json_path=cell_json,
                pim_layer_group_size=args.pim_layer_group_size,
                pim_async=args.pim_async,
                speculative_preload=args.speculative_preload,
            )
            rows.append(row)
            tps = row.get("decode_tokens_per_second")
            tps_s = f"{tps:.4f}" if isinstance(tps, (int, float)) else "—"
            print(f"[sweep]   status={row.get('status')} tps={tps_s} err={row.get('error_type') or ''}", flush=True)

            payload = {
                "model_path": args.model_path,
                "offload_values": args.offload_values,
                "prompt_profiles": args.prompt_profiles,
                "max_new_tokens": args.max_new_tokens,
                "pim_layer_group_size": args.pim_layer_group_size,
                "pim_async": args.pim_async,
                "speculative_preload": args.speculative_preload,
                "results": rows,
                "summary": _summarize(rows),
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"[sweep] wrote {len(rows)} cells to {out_path}")


if __name__ == "__main__":
    main()
