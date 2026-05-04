"""
ADR-002 M-18: profile per-layer expert routing frequency.

Loads the model under a "cheap" backend (CPU or cuda-all-gpu) so PIM
isn't engaged, runs the calibration prompt(s) through prefill +
N decode tokens, and counts how many tokens hit each (layer, expert)
slot via a forward hook on every HybridMoE.forward.  Dumps a
[num_layers, num_experts] frequency table to JSON for later use with
``benchmarks/benchmark_inference.py --routing-freq-json``.

Usage:
    python benchmarks/profile_expert_routing.py \
        --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
        --prompt "Once upon a time" \
        --max-new-tokens 32 \
        --json-out benchmarks/results/routing_freq_qwen3_30b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Make sibling package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_ktrans.llm import LLM  # noqa: E402
from nano_ktrans.layers.hybrid_moe import HybridMoE  # noqa: E402


def _install_router_hooks(
    llm: LLM,
    counts: torch.Tensor,
) -> list:
    """Register a forward-pre-hook on every HybridMoE that increments
    ``counts[layer_idx, expert_idx]`` by the number of tokens routed
    to that expert in this forward call.
    """
    handles: list = []

    def make_hook(lidx: int):
        def hook(module: HybridMoE, args, kwargs):
            # HybridMoE.forward(hidden_states, router_logits)
            # args may be (hidden_states, router_logits) or have
            # router_logits in kwargs depending on call site.
            if len(args) >= 2:
                router_logits = args[1]
            else:
                router_logits = kwargs.get("router_logits", None)
            if router_logits is None or router_logits.numel() == 0:
                return
            top_k = int(module.top_k)
            num_experts = int(module.num_experts)
            # Mirror the topk selection HybridMoE.forward does itself.
            if module.router_use_softmax:
                probs = torch.softmax(
                    router_logits.detach().to(torch.float32), dim=-1
                )
                _, topk_ids = torch.topk(probs, top_k, dim=-1)
            else:
                _, topk_ids = torch.topk(router_logits, top_k, dim=-1)
            # bincount across all tokens this call.
            flat = topk_ids.detach().flatten().to(dtype=torch.long, device="cpu")
            bc = torch.bincount(flat, minlength=num_experts)
            counts[lidx] += bc.to(dtype=counts.dtype)
        return hook

    layer_idx = 0
    for module in llm.model.modules():
        if isinstance(module, HybridMoE):
            handles.append(
                module.register_forward_pre_hook(
                    make_hook(int(module.layer_idx)),
                    with_kwargs=True,
                )
            )
            layer_idx += 1
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADR-002 M-18/M-23: profile per-layer expert routing frequency."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt",
        default="Once upon a time, in a small village by the sea, there lived a "
        "young scholar who spent her evenings cataloguing the names of stars. "
        "Each constellation had its own folklore, and she enjoyed weaving them "
        "into the local children's bedtime stories.",
        help="Single-prompt mode.  Ignored when --prompts-json is provided.",
    )
    parser.add_argument(
        "--prompts-json",
        default=None,
        help=(
            "ADR-002 M-23: batch mode.  Path to a JSON file with a list of "
            "{'name': ..., 'prompt': ...} entries.  One activation_freq file is "
            "dumped per entry into --json-out-dir.  A shared LLM instance is "
            "reused across prompts to amortise the ~220s load cost."
        ),
    )
    parser.add_argument(
        "--json-out-dir",
        default=None,
        help="ADR-002 M-23: output directory for --prompts-json batch mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for calibration.  CPU is fine and avoids CUDA setup; "
        "use cuda when CPU prefill is too slow on a large model.",
    )
    parser.add_argument(
        "--num-gpu-experts",
        type=int,
        default=None,
        help="GPU expert budget during calibration.  Default None = all "
        "experts on the chosen device.  On a 47GB card Qwen3-30B-A3B-GPTQ "
        "does not fit full, so pass e.g. 92 and combine with "
        "--offload-backend pim.",
    )
    parser.add_argument(
        "--offload-backend",
        default="cpu",
        choices=["cpu", "pim"],
        help="Which offload backend to use for experts that do not fit on "
        "the chosen device.  'cpu' is fine but slow on large models; "
        "'pim' lets PIM handle the overflow and keeps calibration runtime "
        "reasonable.  Either way the routing counts are identical because "
        "the router itself sees all experts.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Single-prompt mode output.  Required when --prompts-json is NOT set.",
    )
    args = parser.parse_args()

    # Resolve prompt list up front so we fail fast on bad config.
    batch_mode = args.prompts_json is not None
    if batch_mode:
        if not args.json_out_dir:
            raise SystemExit("--json-out-dir is required with --prompts-json")
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            prompt_entries = json.load(f)
        if not isinstance(prompt_entries, list) or not prompt_entries:
            raise SystemExit(
                "--prompts-json must contain a non-empty list of "
                "{'name': ..., 'prompt': ...} entries"
            )
        for entry in prompt_entries:
            if not isinstance(entry, dict) or "name" not in entry or "prompt" not in entry:
                raise SystemExit(
                    "every --prompts-json entry must have 'name' and 'prompt' fields"
                )
    else:
        if not args.json_out:
            raise SystemExit("--json-out is required in single-prompt mode")
        prompt_entries = [{"name": "default", "prompt": args.prompt}]

    print(f"[profile] loading {args.model_path} on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        args.model_path,
        device=args.device,
        num_gpu_experts=args.num_gpu_experts,
        offload_backend=args.offload_backend,
    )
    load_seconds = time.perf_counter() - t0
    print(f"[profile] loaded in {load_seconds:.2f}s", flush=True)

    config = llm.model.model.config
    num_layers = int(config.num_hidden_layers)
    num_experts = int(config.num_local_experts)

    # Per-prompt loop — reuse the same LLM instance across prompts to
    # amortise the model load cost (~220s on Qwen3-30B-A3B-GPTQ).
    # Install hooks ONCE, reset counts between prompts.
    counts = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    handles = _install_router_hooks(llm, counts)
    if not handles:
        raise SystemExit(
            "No HybridMoE module found in the loaded model — "
            "is this a non-MoE checkpoint?"
        )
    print(f"[profile] hooked {len(handles)} HybridMoE layers", flush=True)

    out_dir = Path(args.json_out_dir) if batch_mode else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict[str, Any]] = []
    for entry in prompt_entries:
        name = str(entry["name"])
        prompt = str(entry["prompt"])
        counts.zero_()
        print(f"[profile] calibrating prompt name={name!r} len={len(prompt)} chars ...", flush=True)
        t1 = time.perf_counter()
        _ = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
        gen_seconds = time.perf_counter() - t1
        total = float(counts.sum().item())
        if total <= 0:
            raise SystemExit(
                f"Routing counts for prompt {name!r} are all zero — hooks did not fire."
            )
        # Per-layer row sum == n_tokens * top_k; normalise to per-layer freq.
        freq = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        payload: dict[str, Any] = {
            "model_path": args.model_path,
            "device": args.device,
            "calibration_prompt_name": name,
            "calibration_prompt": prompt,
            "max_new_tokens": args.max_new_tokens,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "load_seconds": load_seconds,
            "calibration_seconds": gen_seconds,
            "total_routing_decisions": int(total),
            "raw_counts": counts.to(torch.float32).tolist(),
            "activation_freq": freq.to(torch.float32).tolist(),
        }
        if batch_mode:
            assert out_dir is not None
            out_path = out_dir / f"routing_freq_{name}.json"
        else:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[profile] wrote {out_path}  "
            f"(prompt={name!r}, {gen_seconds:.1f}s, "
            f"{num_layers} layers x {num_experts} experts)",
            flush=True,
        )
        all_entries.append({"name": name, "path": str(out_path), "calibration_seconds": gen_seconds})

    for h in handles:
        h.remove()

    if batch_mode:
        assert out_dir is not None
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "device": args.device,
                    "num_gpu_experts": args.num_gpu_experts,
                    "offload_backend": args.offload_backend,
                    "max_new_tokens": args.max_new_tokens,
                    "load_seconds": load_seconds,
                    "entries": all_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[profile] wrote manifest {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
