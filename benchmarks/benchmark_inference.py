#!/usr/bin/env python3
import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from nano_ktrans.llm import LLM
from nano_ktrans.scheduler import (
    SCHEDULER_PROFILE_NAMES,
    normalize_scheduler_profiles,
    summarize_offload_diagnostics,
    summarize_profile_sweep_results,
)


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_single_generation(llm: LLM, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    llm.reset_offload_diagnostics()
    inputs = llm.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(llm.device)
    prompt_tokens = int(input_ids.shape[1])

    llm.engine.start_background_offload_worker()
    try:
        synchronize_if_needed(llm.device)
        total_start = time.perf_counter()

        synchronize_if_needed(llm.device)
        prefill_start = time.perf_counter()
        logits = llm.engine.prefill(input_ids)
        synchronize_if_needed(llm.device)
        prefill_seconds = time.perf_counter() - prefill_start

        next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
        generated_ids = [next_token.item()]
        decode_seconds = 0.0

        for i in range(max_new_tokens - 1):
            synchronize_if_needed(llm.device)
            step_start = time.perf_counter()
            logits = llm.engine.decode_step(next_token, prompt_tokens + i)
            synchronize_if_needed(llm.device)
            decode_seconds += time.perf_counter() - step_start

            next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            generated_ids.append(token_id)
            if token_id == llm.tokenizer.eos_token_id:
                break

        synchronize_if_needed(llm.device)
        total_seconds = time.perf_counter() - total_start
        output_text = llm.tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        llm.engine.stop_background_offload_worker()

    generated_tokens = len(generated_ids)
    decode_tps = generated_tokens / decode_seconds if decode_seconds > 0 else None

    offload_diagnostics = llm.get_offload_diagnostics()
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
        "decode_tokens_per_second": decode_tps,
        "output_text": output_text,
        "scheduler_summary": summarize_offload_diagnostics(offload_diagnostics),
    }
    if llm.device.startswith("cuda") and torch.cuda.is_available():
        result["cuda_max_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    return result

def benchmark_backend(
    *,
    backend: str,
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    cpu_device_experts: int | None,
    cuda_device_experts: int | None,
    offload_device_experts: int | None,
    pim_rank_count: int,
    pim_profile: str,
    pim_max_batch_tokens: int,
    pim_kernel_variant: str,
    pim_prefill_policy: str,
    pim_prefill_token_threshold: int,
    pim_layer_group_size: int,
    pim_enable_speculative_preload_gptq: bool,
    enable_dynamic_expert_scheduler: bool,
    scheduler_prefill_force_gpu_budget_per_layer: int | None,
    scheduler_prefill_collect_only: bool | None,
    scheduler_step_stride_prefill: int | None,
    scheduler_step_stride_decode: int | None,
    scheduler_demotion_idle_steps: int | None,
    scheduler_migration_cooldown_steps: int | None,
    scheduler_decode_require_prefetch_ready: bool | None,
    scheduler_prefetch_candidate_budget_per_layer: int | None,
    scheduler_prepared_cache_budget_per_layer: int | None,
    scheduler_profile: str,
    enable_background_offload_worker: bool,
    background_offload_poll_interval_seconds: float,
) -> dict[str, Any]:
    llm = None
    offload_backend = "cpu"
    offload_backend_kwargs: dict[str, Any] = {}
    if backend == "cpu":
        device = "cpu"
        num_device_experts = cpu_device_experts
    elif backend == "cuda":
        if not torch.cuda.is_available():
            return {"backend": backend, "status": "unavailable", "reason": "torch.cuda.is_available() is false"}
        device = "cuda"
        num_device_experts = cuda_device_experts
    elif backend == "cuda_cpu_offload":
        if not torch.cuda.is_available():
            return {"backend": backend, "status": "unavailable", "reason": "torch.cuda.is_available() is false"}
        device = "cuda"
        num_device_experts = offload_device_experts
    elif backend == "cuda_pim":
        if not torch.cuda.is_available():
            return {"backend": backend, "status": "unavailable", "reason": "torch.cuda.is_available() is false"}
        device = "cuda"
        num_device_experts = offload_device_experts
        offload_backend = "pim"
        offload_backend_kwargs = {
            "pim_rank_count": pim_rank_count,
            "pim_profile": pim_profile,
            "pim_max_batch_tokens": pim_max_batch_tokens,
            "pim_kernel_variant": pim_kernel_variant,
            "pim_prefill_policy": pim_prefill_policy,
            "pim_prefill_token_threshold": pim_prefill_token_threshold,
            "pim_layer_group_size": pim_layer_group_size,
            "enable_speculative_preload_gptq": pim_enable_speculative_preload_gptq,
        }
    elif backend == "cuda_pim_shadow":
        if not torch.cuda.is_available():
            return {"backend": backend, "status": "unavailable", "reason": "torch.cuda.is_available() is false"}
        device = "cuda"
        num_device_experts = offload_device_experts
        offload_backend = "pim_shadow"
        offload_backend_kwargs = {
            "pim_rank_count": pim_rank_count,
            "pim_profile": pim_profile,
            "pim_max_batch_tokens": pim_max_batch_tokens,
            "pim_kernel_variant": pim_kernel_variant,
            "pim_prefill_policy": pim_prefill_policy,
            "pim_prefill_token_threshold": pim_prefill_token_threshold,
            "pim_layer_group_size": pim_layer_group_size,
            "enable_speculative_preload_gptq": pim_enable_speculative_preload_gptq,
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        load_start = time.perf_counter()
        llm = LLM(
            model_path,
            device=device,
            num_gpu_experts=num_device_experts,
            offload_backend=offload_backend,
            offload_backend_kwargs=offload_backend_kwargs,
            enable_dynamic_expert_scheduler=enable_dynamic_expert_scheduler,
            scheduler_prefill_force_gpu_budget_per_layer=scheduler_prefill_force_gpu_budget_per_layer,
            scheduler_prefill_collect_only=scheduler_prefill_collect_only,
            scheduler_step_stride_prefill=scheduler_step_stride_prefill,
            scheduler_step_stride_decode=scheduler_step_stride_decode,
            scheduler_demotion_idle_steps=scheduler_demotion_idle_steps,
            scheduler_migration_cooldown_steps=scheduler_migration_cooldown_steps,
            scheduler_decode_require_prefetch_ready=scheduler_decode_require_prefetch_ready,
            scheduler_prefetch_candidate_budget_per_layer=scheduler_prefetch_candidate_budget_per_layer,
            scheduler_prepared_cache_budget_per_layer=scheduler_prepared_cache_budget_per_layer,
            scheduler_profile=scheduler_profile,
            enable_background_offload_worker=enable_background_offload_worker,
            background_offload_poll_interval_seconds=background_offload_poll_interval_seconds,
        )
        synchronize_if_needed(llm.device)
        load_seconds = time.perf_counter() - load_start

        for _ in range(warmup):
            run_single_generation(llm, prompt, max_new_tokens=max_new_tokens)

        runs = [run_single_generation(llm, prompt, max_new_tokens=max_new_tokens) for _ in range(repeats)]
        offload_diagnostics = llm.get_offload_diagnostics()

        summary = {
            "backend": backend,
            "status": "ok",
            "device": device,
            "num_device_experts": num_device_experts,
            "offload_backend": offload_backend,
            "load_seconds": load_seconds,
            "offload_diagnostics": offload_diagnostics,
            "scheduler_summary": summarize_offload_diagnostics(offload_diagnostics),
            "runs": runs,
        }
        return summary
    except torch.OutOfMemoryError as exc:
        return {
            "backend": backend,
            "status": "oom",
            "device": device,
            "num_device_experts": num_device_experts,
            "offload_backend": offload_backend,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "backend": backend,
            "status": "error",
            "device": device,
            "num_device_experts": num_device_experts,
            "offload_backend": offload_backend,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def default_model_path() -> str:
    candidate = Path("/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base")
    return str(candidate) if candidate.exists() else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nano-ktrans inference backends.")
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help="Local model directory or Hugging Face repo id. Defaults to the local Qwen3 checkpoint when present.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cpu", "cuda", "cuda_cpu_offload"],
        choices=["cpu", "cuda", "cuda_cpu_offload", "cuda_pim", "cuda_pim_shadow"],
        help="Backends to benchmark.",
    )
    parser.add_argument("--prompt", default="请解释一下如何写出结构清晰的 Python 脚本。")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cpu-device-experts",
        type=int,
        default=None,
        help="Experts kept on the active device for the CPU backend. Default keeps all experts on CPU.",
    )
    parser.add_argument(
        "--cuda-device-experts",
        type=int,
        default=None,
        help="Experts kept on the active device for the pure CUDA backend. Default keeps all experts on GPU.",
    )
    parser.add_argument(
        "--offload-device-experts",
        type=int,
        default=2,
        help="Experts kept on GPU for the CUDA+CPU-offload backend.",
    )
    parser.add_argument(
        "--pim-rank-count",
        type=int,
        default=1,
        help="Visible PIM ranks to report when running PIM backends.",
    )
    parser.add_argument(
        "--pim-profile",
        default="",
        help="Optional libdpu allocation profile passed to the real PIM backend.",
    )
    parser.add_argument(
        "--pim-max-batch-tokens",
        type=int,
        default=1,
        help="Maximum flattened token rows routed through the real PIM backend before falling back to CPU.",
    )
    parser.add_argument(
        "--pim-kernel-variant",
        default="linear",
        choices=["linear", "fused"],
        help="Real PIM kernel variant: 'linear' runs three DPU linears with host activation, 'fused' runs the full expert MLP on DPU.",
    )
    parser.add_argument(
        "--pim-prefill-policy",
        default="cpu",
        choices=["cpu", "pim"],
        help="Prefill policy for real PIM backend. Recommended: keep prefill on CPU/GPU.",
    )
    parser.add_argument(
        "--pim-prefill-token-threshold",
        type=int,
        default=8,
        help="Maximum flattened prefill tokens allowed to use real PIM before forcing fallback.",
    )
    # ADR-002 M-9: expose the M-7 layer-group scoping size + the M-7
    # speculative preload flag (kept default-on after M-8's handle fix).
    parser.add_argument(
        "--pim-layer-group-size",
        type=int,
        default=48,
        help=(
            "ADR-002 M-7/M-8/M-9: how many MoE layers share one DPU runtime pair. "
            "Default 48 = all layers share one runtime pair (M-6 equivalent); M-9 "
            "real-hardware sweep showed this is the fastest configuration because "
            "32-rank-pool coordination overhead exceeds multi-slot hit benefit "
            "when Qwen3 routing locality is ~14%% (ADR-002 §17). "
            "group_size=1 => every layer has its own rank pool; group_size=3 was "
            "the M-7/M-8 default, now superseded."
        ),
    )
    parser.add_argument(
        "--pim-enable-speculative-preload-gptq",
        dest="pim_enable_speculative_preload_gptq",
        action="store_true",
        default=False,
        help=(
            "ADR-002 M-7: warm the MRAM slot LRU at prefill end with each layer's "
            "top-N hot experts.  Default OFF after M-9 showed the preload/hit ratio "
            "(24/96 hits in M-8) does not compensate for the extra prefill time."
        ),
    )
    parser.add_argument(
        "--no-pim-speculative-preload-gptq",
        dest="pim_enable_speculative_preload_gptq",
        action="store_false",
        help="Disable the M-7 speculative preload (useful for A/B benchmarking).",
    )
    parser.add_argument(
        "--enable-dynamic-expert-scheduler",
        action="store_true",
        help="Enable experimental dynamic GPU/PIM expert residency scheduler.",
    )
    parser.add_argument(
        "--scheduler-prefill-force-gpu-budget-per-layer",
        type=int,
        default=None,
        help="During prefill, temporarily target at least this many GPU-resident experts per layer.",
    )
    parser.add_argument(
        "--scheduler-prefill-collect-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="During prefill, only collect hotness and prefetch candidates without emitting migrations.",
    )
    parser.add_argument("--scheduler-step-stride-prefill", type=int, default=None)
    parser.add_argument("--scheduler-step-stride-decode", type=int, default=None)
    parser.add_argument("--scheduler-demotion-idle-steps", type=int, default=None)
    parser.add_argument("--scheduler-migration-cooldown-steps", type=int, default=None)
    parser.add_argument(
        "--scheduler-decode-require-prefetch-ready",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--scheduler-prefetch-candidate-budget-per-layer", type=int, default=None)
    parser.add_argument("--scheduler-prepared-cache-budget-per-layer", type=int, default=None)
    parser.add_argument(
        "--scheduler-profile",
        default="baseline",
        choices=list(SCHEDULER_PROFILE_NAMES),
        help="Scheduler preset for dynamic expert migration experiments.",
    )
    parser.add_argument(
        "--enable-background-offload-worker",
        action="store_true",
        help="Enable the experimental background offload worker during benchmarked generation runs.",
    )
    parser.add_argument(
        "--background-offload-poll-interval-seconds",
        type=float,
        default=0.005,
        help="Polling interval in seconds for the experimental background offload worker.",
    )
    parser.add_argument(
        "--scheduler-profile-sweep",
        nargs="+",
        choices=list(SCHEDULER_PROFILE_NAMES),
        help="Optional list of scheduler profiles to benchmark in one run.",
    )
    parser.add_argument("--json-out", help="Optional path to write the benchmark results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path is required when the local default checkpoint is absent.")

    results = {
        "model_path": args.model_path,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results": [],
    }

    profile_list = normalize_scheduler_profiles(
        args.scheduler_profile_sweep,
        default_profile=args.scheduler_profile,
    )
    results["scheduler_profiles"] = profile_list

    for scheduler_profile in profile_list:
        for backend in args.backends:
            result = benchmark_backend(
                backend=backend,
                model_path=args.model_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                repeats=args.repeats,
                cpu_device_experts=args.cpu_device_experts,
                cuda_device_experts=args.cuda_device_experts,
                offload_device_experts=args.offload_device_experts,
                pim_rank_count=args.pim_rank_count,
                pim_profile=args.pim_profile,
                pim_max_batch_tokens=args.pim_max_batch_tokens,
                pim_kernel_variant=args.pim_kernel_variant,
                pim_prefill_policy=args.pim_prefill_policy,
                pim_prefill_token_threshold=args.pim_prefill_token_threshold,
                pim_layer_group_size=args.pim_layer_group_size,
                pim_enable_speculative_preload_gptq=args.pim_enable_speculative_preload_gptq,
                enable_dynamic_expert_scheduler=args.enable_dynamic_expert_scheduler,
                scheduler_prefill_force_gpu_budget_per_layer=args.scheduler_prefill_force_gpu_budget_per_layer,
                scheduler_prefill_collect_only=args.scheduler_prefill_collect_only,
                scheduler_step_stride_prefill=args.scheduler_step_stride_prefill,
                scheduler_step_stride_decode=args.scheduler_step_stride_decode,
                scheduler_demotion_idle_steps=args.scheduler_demotion_idle_steps,
                scheduler_migration_cooldown_steps=args.scheduler_migration_cooldown_steps,
                scheduler_decode_require_prefetch_ready=args.scheduler_decode_require_prefetch_ready,
                scheduler_prefetch_candidate_budget_per_layer=args.scheduler_prefetch_candidate_budget_per_layer,
                scheduler_prepared_cache_budget_per_layer=args.scheduler_prepared_cache_budget_per_layer,
                scheduler_profile=scheduler_profile,
                enable_background_offload_worker=args.enable_background_offload_worker,
                background_offload_poll_interval_seconds=args.background_offload_poll_interval_seconds,
            )
            result["scheduler_profile"] = scheduler_profile
            results["results"].append(result)

    results["profile_sweep_summary"] = summarize_profile_sweep_results(results["results"])

    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)

    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
