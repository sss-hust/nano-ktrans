#!/usr/bin/env python3
import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from nano_ktrans.llm import LLM


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_single_generation(llm: LLM, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    inputs = llm.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(llm.device)
    prompt_tokens = int(input_ids.shape[1])

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

    generated_tokens = len(generated_ids)
    decode_tps = generated_tokens / decode_seconds if decode_seconds > 0 else None

    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
        "decode_tokens_per_second": decode_tps,
        "output_text": output_text,
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
    elif backend == "cuda_pim_shadow":
        if not torch.cuda.is_available():
            return {"backend": backend, "status": "unavailable", "reason": "torch.cuda.is_available() is false"}
        device = "cuda"
        num_device_experts = offload_device_experts
        offload_backend = "pim_shadow"
        offload_backend_kwargs = {"pim_rank_count": pim_rank_count}
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
        )
        synchronize_if_needed(llm.device)
        load_seconds = time.perf_counter() - load_start

        for _ in range(warmup):
            run_single_generation(llm, prompt, max_new_tokens=max_new_tokens)

        runs = [run_single_generation(llm, prompt, max_new_tokens=max_new_tokens) for _ in range(repeats)]

        summary = {
            "backend": backend,
            "status": "ok",
            "device": device,
            "num_device_experts": num_device_experts,
            "offload_backend": offload_backend,
            "load_seconds": load_seconds,
            "offload_diagnostics": llm.get_offload_diagnostics(),
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
        choices=["cpu", "cuda", "cuda_cpu_offload", "cuda_pim_shadow"],
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
        help="Visible PIM ranks to report when running the experimental cuda_pim_shadow backend.",
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
        )
        results["results"].append(result)

    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)

    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
