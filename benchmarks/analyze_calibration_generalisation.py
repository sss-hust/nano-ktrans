"""
ADR-002 M-23: quantify the generalisation gap of offline calibration.

Reads multiple activation_freq JSON files produced by
benchmarks/profile_expert_routing.py and computes:

  1. Per-prompt hottest-N mask overlap (static view).  If the masks
     are ~identical across prompts, dynamic scheduling's theoretical
     headroom is close to zero.
  2. Cross-prompt PIM routing share (dynamic view).  If calibration
     on prompt A gives PIM share ~ X% on prompt A but >> X% when
     applied to prompt B, static M-18 does NOT generalise.
  3. First-N (uniform) baseline as the worst-case reference.

Outputs a compact JSON summary suitable for ADR artefact + a human
readable table on stdout.

Usage:
    python benchmarks/analyze_calibration_generalisation.py \
        --calibration-dir benchmarks/results/m23_calibrations \
        --num-gpu-experts 92 \
        --json-out benchmarks/results/m23_calibration_generalisation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_calibrations(calibration_dir: Path) -> dict[str, torch.Tensor]:
    """Load every routing_freq_*.json (skipping manifest.json) under
    ``calibration_dir`` and return {name: freq_tensor [L, E]}."""
    entries: dict[str, torch.Tensor] = {}
    for path in sorted(calibration_dir.glob("routing_freq_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("calibration_prompt_name") or path.stem
        freq = torch.tensor(data["activation_freq"], dtype=torch.float32)
        if freq.ndim != 2:
            raise ValueError(f"{path} activation_freq is not 2D")
        entries[str(name)] = freq
    if not entries:
        raise SystemExit(f"no routing_freq_*.json files found under {calibration_dir}")
    shape0 = next(iter(entries.values())).shape
    for name, freq in entries.items():
        if freq.shape != shape0:
            raise ValueError(f"{name} shape {tuple(freq.shape)} != {tuple(shape0)}")
    return entries


def _hottest_mask(freq: torch.Tensor, k: int) -> torch.Tensor:
    """Return a bool mask [L, E] with True on the top-k experts per layer."""
    _, top_idx = torch.topk(freq, k=k, dim=1)
    mask = torch.zeros_like(freq, dtype=torch.bool)
    mask.scatter_(1, top_idx, True)
    return mask


def _mask_overlap_per_layer(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """Per-layer intersection size / union size (Jaccard).  Since both
    masks have the same cardinality k, |A ∩ B| / |A ∪ B| reduces to
    |A ∩ B| / (2k - |A ∩ B|).  We return the raw intersection count
    and jaccard separately for readability."""
    intersection = (mask_a & mask_b).sum(dim=1).to(torch.int64)
    return intersection


def _pim_share(freq: torch.Tensor, gpu_mask: torch.Tensor) -> torch.Tensor:
    """Per-layer fraction of routing mass that lands on CPU/PIM experts
    when ``gpu_mask`` is applied to this prompt's routing freq."""
    gpu_mass = (freq * gpu_mask.to(freq.dtype)).sum(dim=1)
    # per-layer row sum is normalised to 1 by profile_expert_routing.py
    return 1.0 - gpu_mass


def _first_n_mask(num_layers: int, num_experts: int, k: int) -> torch.Tensor:
    mask = torch.zeros(num_layers, num_experts, dtype=torch.bool)
    mask[:, :k] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADR-002 M-23: analyse cross-prompt generalisation of calibration masks."
    )
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument(
        "--num-gpu-experts",
        type=int,
        default=92,
        help="GPU expert budget to evaluate (default 92, matches M-18).",
    )
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    calib_dir = Path(args.calibration_dir)
    freqs = _load_calibrations(calib_dir)
    names = list(freqs.keys())
    num_layers, num_experts = next(iter(freqs.values())).shape
    k = min(args.num_gpu_experts, num_experts)

    # Per-prompt hottest-k masks.
    masks = {name: _hottest_mask(freq, k) for name, freq in freqs.items()}

    # Uniform baseline mask (first-N).
    first_n = _first_n_mask(num_layers, num_experts, k)

    # 1. Pairwise mask overlap (static view).
    overlap_matrix: dict[str, dict[str, dict[str, float]]] = {}
    for a in names:
        row: dict[str, dict[str, float]] = {}
        for b in names:
            inter = _mask_overlap_per_layer(masks[a], masks[b])
            # Per-layer intersection has max = k.  Fraction = mean(inter) / k.
            mean_inter = float(inter.to(torch.float32).mean().item())
            min_inter = int(inter.min().item())
            max_inter = int(inter.max().item())
            row[b] = {
                "mean_intersection_per_layer": mean_inter,
                "mean_intersection_fraction": mean_inter / k if k > 0 else 0.0,
                "min_intersection": min_inter,
                "max_intersection": max_inter,
                "k": k,
            }
        overlap_matrix[a] = row

    # Also vs first-N baseline.
    first_n_overlap: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        inter = _mask_overlap_per_layer(mask, first_n)
        mean_inter = float(inter.to(torch.float32).mean().item())
        first_n_overlap[name] = {
            "mean_intersection_per_layer": mean_inter,
            "mean_intersection_fraction": mean_inter / k if k > 0 else 0.0,
            "min_intersection": int(inter.min().item()),
            "max_intersection": int(inter.max().item()),
            "k": k,
        }

    # 2. Cross-prompt PIM share (dynamic view).
    #    pim_share[calib][eval] = PIM share when calibrating on `calib`
    #    and running on `eval`.  Diagonal is self-calibration share.
    pim_share_matrix: dict[str, dict[str, float]] = {}
    for calib in names:
        cmask = masks[calib]
        row: dict[str, float] = {}
        for ev in names:
            pim_share_layer = _pim_share(freqs[ev], cmask)
            row[ev] = float(pim_share_layer.mean().item())
        pim_share_matrix[calib] = row

    # First-N baseline PIM share on each prompt.
    first_n_pim_share = {
        ev: float(_pim_share(freqs[ev], first_n).mean().item()) for ev in names
    }

    # ADR-002 M-23.1: "general-purpose" mask built from the mean of
    # all calibration freqs.  If this evaluates almost as well as
    # self-calibration on every held-out prompt, static multi-prompt
    # calibration is a simpler alternative to dynamic scheduling.
    mean_freq = torch.stack(list(freqs.values())).mean(dim=0)
    mean_mask = _hottest_mask(mean_freq, k)
    mean_mask_pim_share = {
        ev: float(_pim_share(freqs[ev], mean_mask).mean().item()) for ev in names
    }
    # Per-prompt overlap of the mean mask vs self-calibration mask.
    mean_mask_overlap: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        inter = _mask_overlap_per_layer(mean_mask, mask)
        mean_inter = float(inter.to(torch.float32).mean().item())
        mean_mask_overlap[name] = {
            "mean_intersection_per_layer": mean_inter,
            "mean_intersection_fraction": mean_inter / k if k > 0 else 0.0,
            "min_intersection": int(inter.min().item()),
            "max_intersection": int(inter.max().item()),
            "k": k,
        }

    # 3. Summary stats to foreground the decision.
    self_shares = [pim_share_matrix[n][n] for n in names]
    cross_shares = [
        pim_share_matrix[a][b] for a in names for b in names if a != b
    ]
    self_mean = sum(self_shares) / len(self_shares)
    cross_mean = sum(cross_shares) / len(cross_shares) if cross_shares else self_mean
    first_n_mean = sum(first_n_pim_share.values()) / len(first_n_pim_share)
    mean_mask_mean = sum(mean_mask_pim_share.values()) / len(mean_mask_pim_share)

    summary = {
        "self_calib_pim_share_mean": self_mean,
        "cross_calib_pim_share_mean": cross_mean,
        "mean_mask_pim_share_mean": mean_mask_mean,
        "first_n_pim_share_mean": first_n_mean,
        "generalisation_gap_pp": (cross_mean - self_mean) * 100.0,
        "vs_uniform_improvement_static_pp": (first_n_mean - cross_mean) * 100.0,
        "mean_mask_vs_cross_calib_pp": (cross_mean - mean_mask_mean) * 100.0,
        "mean_mask_vs_self_calib_pp": (mean_mask_mean - self_mean) * 100.0,
        "k": k,
        "num_prompts": len(names),
        "num_layers": num_layers,
        "num_experts": num_experts,
    }

    # Pretty-print the key tables.
    print()
    print(f"=== M-23 calibration generalisation (k={k}, {len(names)} prompts) ===")
    print()
    max_name_len = max(len(n) for n in names + ["calib\\eval"])
    col_w = max(max_name_len + 2, 12)
    print("Pairwise mask overlap (mean intersection / layer, max=k):")
    header = "calib\\eval".ljust(col_w) + "".join(n.rjust(col_w) for n in names)
    print(header)
    for a in names:
        cells = "".join(
            f"{overlap_matrix[a][b]['mean_intersection_per_layer']:>{col_w}.2f}" for b in names
        )
        print(f"{a:<{col_w}}{cells}")
    print()
    print(f"First-N baseline vs each calibration mask (overlap / {k}):")
    for name, row in first_n_overlap.items():
        print(
            f"  {name:<{col_w}}  mean={row['mean_intersection_per_layer']:>6.2f}  "
            f"({row['mean_intersection_fraction']*100:.1f}%)"
        )
    print()
    print("Cross-prompt PIM routing share matrix (% of routing to PIM):")
    header = "calib\\eval".ljust(col_w) + "".join(n.rjust(col_w) for n in names)
    print(header)
    for a in names:
        cells = "".join(f"{pim_share_matrix[a][b]*100:>{col_w}.2f}" for b in names)
        print(f"{a:<{col_w}}{cells}")
    print()
    print("First-N uniform baseline PIM share per prompt (%):")
    for ev, v in first_n_pim_share.items():
        print(f"  {ev:<{col_w}}  {v*100:.2f}%")
    print()
    print("M-23.1 mean-mask (averaged freq, hottest-k) PIM share per prompt (%):")
    for ev, v in mean_mask_pim_share.items():
        print(f"  {ev:<{col_w}}  {v*100:.2f}%")
    print()
    print(f"Mean-mask vs each self-calib mask (overlap / {k}):")
    for name, row in mean_mask_overlap.items():
        print(
            f"  {name:<{col_w}}  mean={row['mean_intersection_per_layer']:>6.2f}  "
            f"({row['mean_intersection_fraction']*100:.1f}%)"
        )
    print()
    print(
        "Summary:"
        f"\n  self-calibration PIM share mean : {self_mean*100:.2f}%"
        f"\n  cross-calibration PIM share mean: {cross_mean*100:.2f}%"
        f"\n  mean-mask PIM share mean        : {mean_mask_mean*100:.2f}%"
        f"\n  first-N uniform PIM share mean  : {first_n_mean*100:.2f}%"
        f"\n  generalisation gap (cross-self) : {summary['generalisation_gap_pp']:+.2f} pp"
        f"\n  mean-mask wins over cross-calib : {summary['mean_mask_vs_cross_calib_pp']:+.2f} pp"
        f"\n  mean-mask loses to self-calib   : {summary['mean_mask_vs_self_calib_pp']:+.2f} pp"
        f"\n  static wins over uniform        : {summary['vs_uniform_improvement_static_pp']:+.2f} pp"
    )
    print()

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "calibration_dir": str(calib_dir),
                "num_gpu_experts": args.num_gpu_experts,
                "prompts": names,
                "first_n_overlap": first_n_overlap,
                "mask_overlap_matrix": overlap_matrix,
                "pim_share_matrix": pim_share_matrix,
                "first_n_pim_share": first_n_pim_share,
                "mean_mask_pim_share": mean_mask_pim_share,
                "mean_mask_overlap_vs_self": mean_mask_overlap,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[M-23] wrote {out}")


if __name__ == "__main__":
    main()
