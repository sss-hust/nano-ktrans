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
        description="ADR-002 M-18: profile per-layer expert routing frequency."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt",
        default="Once upon a time, in a small village by the sea, there lived a "
        "young scholar who spent her evenings cataloguing the names of stars. "
        "Each constellation had its own folklore, and she enjoyed weaving them "
        "into the local children's bedtime stories.",
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
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

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
    counts = torch.zeros(num_layers, num_experts, dtype=torch.float64)

    handles = _install_router_hooks(llm, counts)
    if not handles:
        raise SystemExit(
            "No HybridMoE module found in the loaded model — "
            "is this a non-MoE checkpoint?"
        )
    print(f"[profile] hooked {len(handles)} HybridMoE layers", flush=True)

    print(f"[profile] running calibration generate ...", flush=True)
    t1 = time.perf_counter()
    # LLM.generate() re-tokenises a str prompt itself; just pass it through.
    _ = llm.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    gen_seconds = time.perf_counter() - t1
    print(f"[profile] calibration done in {gen_seconds:.2f}s", flush=True)

    for h in handles:
        h.remove()

    total = float(counts.sum().item())
    if total <= 0:
        raise SystemExit("Routing counts are all zero — hooks did not fire.")
    # Normalise to per-token frequency per layer.  Each token activates
    # top_k experts per layer, so per-layer row sum == n_tokens * top_k.
    freq = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)

    payload: dict[str, Any] = {
        "model_path": args.model_path,
        "device": args.device,
        "calibration_prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "load_seconds": load_seconds,
        "calibration_seconds": gen_seconds,
        "total_routing_decisions": int(total),
        "raw_counts": counts.to(torch.float32).tolist(),
        "activation_freq": freq.to(torch.float32).tolist(),
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[profile] wrote {out}  ({num_layers} layers x {num_experts} experts)")


if __name__ == "__main__":
    main()
