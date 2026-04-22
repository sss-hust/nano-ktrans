"""
BackendCostModel — ADR-002 M-3 routing primitive.

The cost model answers one question per MoE layer call:

    Given (expert shape, batch, rank_count, prefill?), is it cheaper to
    run this layer's CPU-side experts on PIM or on CPU-AMX/grouped?

It is **data-driven**: the initial table is M-2's 120-cell operator sweep
(``cost_model_baseline_m2.json``, auto-generated from
``benchmarks/results/pim_shape_sweep_M2_tmac.json``).  Only
``kernel_mode=4`` rows feed PIM predictions, because ADR-002 §10
established ``kernel_mode=7`` as a negative result.

The runtime keeps per-(shape, batch, rank, backend) EMA so that real
observations gradually correct the baseline.  Prediction lookup falls
back to the nearest neighbour on rank, then on batch, then to an
infinite cost (= never pick that backend).

Zero third-party deps.  Pure Python / stdlib / torch-free.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "BackendCostModel",
    "BackendDecision",
    "CostEstimate",
    "load_default_cost_model",
]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Per-backend predicted seconds-per-call plus how we got the number."""

    seconds: float
    source: str  # "exact" | "nearest_rank" | "nearest_batch" | "fallback" | "updated_ema"
    ratio_vs_cpu: Optional[float] = None


@dataclass
class BackendDecision:
    """Result of :meth:`BackendCostModel.decide`."""

    backend: str  # "pim" | "cpu"
    pim_seconds: Optional[float]
    cpu_seconds: Optional[float]
    reason: str  # human-readable justification
    source: str  # how the estimate was derived


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    pim_seconds_avg: float
    cpu_grouped_seconds_avg: float
    samples: int = 1


class BackendCostModel:
    """
    Cost-based backend routing for MoE expert calls.

    Key design decisions (from ADR-002 M-3):

    1. **Shape bucketing.** We bucket by the projection label
       (``gate``/``up``/``down``), NOT by raw ``(in_features, out_features)``.
       This matches how the M-2 sweep was run on Qwen3-30B-A3B-GPTQ-Int4:
       gate/up are ``2048 -> 768``, down is ``768 -> 2048``.  Other
       models' MoE experts get routed by the best-matching shape label
       (falling back to nearest-neighbour on raw dims).

    2. **Batch bucketing.** Exact match on the {1, 2, 4, 8} bucket the
       sweep uses.  Unknown batch sizes get the closest swept batch.
       Callers should clamp their "virtual batch" (= tokens routed to
       the CPU-side experts this layer) to the same axis.

    3. **Rank fallback.** UPMEM rank_count is a hardware property of the
       PIM runtime; we do **not** vary it per decision.  We pick the
       nearest-rank cell from the table at construction time.

    4. **EMA update.** Post-call observations update the in-memory table
       with an exponential moving average (``alpha=0.25``).  This lets
       the runtime correct for data-distribution drift or different
       hardware without regenerating the baseline.

    5. **"Never downgrade a known winner."** If both backends have an
       estimate and their ratio is <= ``stability_margin`` (default 1.1),
       we keep the previously-picked backend for this (shape, batch) —
       prevents thrashing when PIM and CPU are near-parity.
    """

    # Shape class mapping — matches QWEN3_EXPERT_SHAPES in the sweep.
    _KNOWN_SHAPES: tuple[str, ...] = ("gate", "up", "down")

    DEFAULT_EMA_ALPHA: float = 0.25
    DEFAULT_STABILITY_MARGIN: float = 1.1

    def __init__(
        self,
        *,
        table: Optional[list[dict[str, Any]]] = None,
        source_path: Optional[str] = None,
        kernel_mode: int = 4,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        stability_margin: float = DEFAULT_STABILITY_MARGIN,
        default_rank_count: int = 1,
    ) -> None:
        self._ema_alpha = float(ema_alpha)
        self._stability_margin = float(stability_margin)
        self._kernel_mode = int(kernel_mode)
        self._source_path = source_path
        self._default_rank_count = int(default_rank_count)
        self._lock = threading.Lock()

        # Stores: {(shape_name, batch, rank): _Cell}
        self._cells: dict[tuple[str, int, int], _Cell] = {}
        # Stores: {(shape_name, batch): last_backend} — used for stability_margin.
        self._last_pick: dict[tuple[str, int], str] = {}

        # Decision counters (diagnostics).
        self.decisions_pim: int = 0
        self.decisions_cpu: int = 0
        self.ema_updates: int = 0

        # Derived shape metadata from the table (for shape-by-dim fallback).
        # {(in_features, out_features): shape_name}
        self._dim_to_shape: dict[tuple[int, int], str] = {}
        # {shape_name: (in_features, out_features)}
        self._shape_to_dim: dict[str, tuple[int, int]] = {}

        if table is not None:
            self._load_table(table)

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "BackendCostModel":
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)
        kernel_mode = int(payload.get("kernel_mode", 4))
        table = payload.get("table", [])
        return cls(
            table=table,
            source_path=str(path),
            kernel_mode=kernel_mode,
            **kwargs,
        )

    def _load_table(self, table: list[dict[str, Any]]) -> None:
        for row in table:
            shape = str(row["shape_name"])
            batch = int(row["batch"])
            rank = int(row["rank_count"])
            in_features = int(row["in_features"])
            out_features = int(row["out_features"])
            cell = _Cell(
                pim_seconds_avg=float(row["pim_seconds_avg"]),
                cpu_grouped_seconds_avg=float(row["cpu_grouped_seconds_avg"]),
                samples=1,
            )
            self._cells[(shape, batch, rank)] = cell
            self._dim_to_shape[(in_features, out_features)] = shape
            self._shape_to_dim[shape] = (in_features, out_features)

    # -------------------------------------------------------------- Lookup

    def _resolve_shape(
        self,
        shape_name: Optional[str],
        in_features: Optional[int],
        out_features: Optional[int],
    ) -> Optional[str]:
        if shape_name is not None and shape_name in self._shape_to_dim:
            return shape_name
        if in_features is not None and out_features is not None:
            direct = self._dim_to_shape.get((in_features, out_features))
            if direct is not None:
                return direct
            # Nearest-neighbour by (in, out) ratio.
            best: Optional[str] = None
            best_dist = math.inf
            target = float(in_features) / max(float(out_features), 1.0)
            for (in_f, out_f), name in self._dim_to_shape.items():
                r = float(in_f) / max(float(out_f), 1.0)
                dist = abs(math.log(r) - math.log(target))
                if dist < best_dist:
                    best_dist = dist
                    best = name
            return best
        # Totally unknown.
        return None

    def _nearest_rank(self, shape: str, batch: int, rank: int) -> Optional[int]:
        ranks = [r for (s, b, r) in self._cells if s == shape and b == batch]
        if not ranks:
            return None
        return min(ranks, key=lambda r: abs(r - rank))

    def _nearest_batch(self, shape: str, batch: int, rank: int) -> Optional[tuple[int, int]]:
        # Step 1: best rank for any batch.
        candidates = [(b, r) for (s, b, r) in self._cells if s == shape]
        if not candidates:
            return None
        # Prefer exact rank then nearest batch; else nearest (batch, rank) lex.
        exact_rank = [b for (b, r) in candidates if r == rank]
        if exact_rank:
            nearest_b = min(exact_rank, key=lambda b: abs(b - batch))
            return (nearest_b, rank)
        return min(candidates, key=lambda br: (abs(br[0] - batch), abs(br[1] - rank)))

    def estimate(
        self,
        *,
        shape_name: Optional[str] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        batch: int,
        rank_count: Optional[int] = None,
    ) -> tuple[Optional[CostEstimate], Optional[CostEstimate]]:
        """
        Return ``(pim_estimate, cpu_estimate)`` for the given cell.

        Either estimate may be ``None`` if the cost model has no data
        at all for the shape bucket.
        """
        shape = self._resolve_shape(shape_name, in_features, out_features)
        if shape is None:
            return None, None
        rank = int(rank_count) if rank_count is not None else self._default_rank_count

        with self._lock:
            cell = self._cells.get((shape, batch, rank))
            if cell is not None:
                return self._cell_to_estimates(cell, source="exact")

            # Try nearest rank at the same batch.
            near_rank = self._nearest_rank(shape, batch, rank)
            if near_rank is not None:
                cell = self._cells[(shape, batch, near_rank)]
                return self._cell_to_estimates(cell, source="nearest_rank")

            # Try nearest batch.
            nb = self._nearest_batch(shape, batch, rank)
            if nb is not None:
                b, r = nb
                cell = self._cells[(shape, b, r)]
                return self._cell_to_estimates(cell, source="nearest_batch")

        return None, None

    @staticmethod
    def _cell_to_estimates(
        cell: _Cell, source: str
    ) -> tuple[CostEstimate, CostEstimate]:
        ratio = (
            cell.cpu_grouped_seconds_avg / cell.pim_seconds_avg
            if cell.pim_seconds_avg > 0
            else None
        )
        pim = CostEstimate(
            seconds=cell.pim_seconds_avg,
            source=source,
            ratio_vs_cpu=ratio,
        )
        cpu = CostEstimate(
            seconds=cell.cpu_grouped_seconds_avg,
            source=source,
            ratio_vs_cpu=(1.0 / ratio) if ratio else None,
        )
        return pim, cpu

    # ------------------------------------------------------------ Decision

    def decide(
        self,
        *,
        shape_name: Optional[str] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        batch: int,
        rank_count: Optional[int] = None,
        is_prefill: bool = False,
        pim_available: bool = True,
    ) -> BackendDecision:
        """
        Return the recommended backend for this expert-layer call.

        Semantics:

        * If ``pim_available`` is False, always pick "cpu".
        * If cost model has no data for the shape, default to "cpu"
          (honest fallback — we do not gamble).
        * If both estimates exist, pick the smaller.  Apply a stability
          margin on re-decisions for the same (shape, batch) pair so
          small predicted differences don't cause backend thrashing.
        * ``is_prefill`` is recorded in the decision reason; the raw
          measurement already reflects how PIM behaves at large batch,
          so we trust the numbers rather than a separate prefill rule.
        """
        if not pim_available:
            self.decisions_cpu += 1
            return BackendDecision(
                backend="cpu",
                pim_seconds=None,
                cpu_seconds=None,
                reason="pim_unavailable",
                source="rule",
            )

        pim_est, cpu_est = self.estimate(
            shape_name=shape_name,
            in_features=in_features,
            out_features=out_features,
            batch=batch,
            rank_count=rank_count,
        )

        shape = self._resolve_shape(shape_name, in_features, out_features) or "unknown"
        key = (shape, int(batch))

        if pim_est is None or cpu_est is None:
            self.decisions_cpu += 1
            return BackendDecision(
                backend="cpu",
                pim_seconds=pim_est.seconds if pim_est else None,
                cpu_seconds=cpu_est.seconds if cpu_est else None,
                reason="no_data_for_shape_bucket",
                source=pim_est.source if pim_est else (cpu_est.source if cpu_est else "fallback"),
            )

        raw_choice = "pim" if pim_est.seconds <= cpu_est.seconds else "cpu"
        source = pim_est.source

        # Stability margin.
        with self._lock:
            previous = self._last_pick.get(key)

        if previous is not None and raw_choice != previous:
            ratio = (
                max(pim_est.seconds, cpu_est.seconds)
                / max(min(pim_est.seconds, cpu_est.seconds), 1e-12)
            )
            if ratio < self._stability_margin:
                raw_choice = previous
                source = f"{source}+stability_margin"

        if is_prefill:
            reason = (
                f"prefill: cost_model picked {raw_choice}"
                f" (pim={pim_est.seconds:.4g}s vs cpu={cpu_est.seconds:.4g}s,"
                f" source={pim_est.source})"
            )
        else:
            reason = (
                f"decode: cost_model picked {raw_choice}"
                f" (pim={pim_est.seconds:.4g}s vs cpu={cpu_est.seconds:.4g}s,"
                f" source={pim_est.source})"
            )

        with self._lock:
            self._last_pick[key] = raw_choice
            if raw_choice == "pim":
                self.decisions_pim += 1
            else:
                self.decisions_cpu += 1

        return BackendDecision(
            backend=raw_choice,
            pim_seconds=pim_est.seconds,
            cpu_seconds=cpu_est.seconds,
            reason=reason,
            source=source,
        )

    # --------------------------------------------------------------- EMA

    def update(
        self,
        *,
        shape_name: Optional[str] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        batch: int,
        rank_count: Optional[int] = None,
        backend: str,
        observed_seconds: float,
    ) -> None:
        """
        Fold a real observation into the cost model via EMA.

        Called from the runtime after a forward pass finishes: we
        average the new seconds-per-call into the matching cell so the
        next decision is informed by live data.
        """
        if observed_seconds <= 0 or not math.isfinite(observed_seconds):
            return
        shape = self._resolve_shape(shape_name, in_features, out_features)
        if shape is None:
            return
        rank = int(rank_count) if rank_count is not None else self._default_rank_count
        key = (shape, int(batch), rank)
        alpha = self._ema_alpha

        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                # Bootstrap a new cell from this single observation.
                if backend == "pim":
                    cell = _Cell(
                        pim_seconds_avg=observed_seconds,
                        cpu_grouped_seconds_avg=observed_seconds,
                        samples=1,
                    )
                elif backend == "cpu":
                    cell = _Cell(
                        pim_seconds_avg=observed_seconds,
                        cpu_grouped_seconds_avg=observed_seconds,
                        samples=1,
                    )
                else:
                    return
                self._cells[key] = cell
                self.ema_updates += 1
                return

            if backend == "pim":
                cell.pim_seconds_avg = (
                    (1 - alpha) * cell.pim_seconds_avg + alpha * observed_seconds
                )
            elif backend == "cpu":
                cell.cpu_grouped_seconds_avg = (
                    (1 - alpha) * cell.cpu_grouped_seconds_avg + alpha * observed_seconds
                )
            else:
                return
            cell.samples += 1
            self.ema_updates += 1

    # ---------------------------------------------------------- Diagnostics

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_path": self._source_path,
                "kernel_mode": self._kernel_mode,
                "cell_count": len(self._cells),
                "shape_count": len(self._shape_to_dim),
                "ema_alpha": self._ema_alpha,
                "stability_margin": self._stability_margin,
                "default_rank_count": self._default_rank_count,
                "decisions_pim": self.decisions_pim,
                "decisions_cpu": self.decisions_cpu,
                "ema_updates": self.ema_updates,
            }


# ---------------------------------------------------------------------------
# Default loader
# ---------------------------------------------------------------------------


_DEFAULT_BASELINE_PATH = Path(__file__).with_name("cost_model_baseline_m2.json")


def load_default_cost_model(
    *,
    default_rank_count: int = 1,
    **kwargs: Any,
) -> Optional["BackendCostModel"]:
    """
    Return a cost model pre-loaded from the M-2 baseline, or ``None``
    if that file is missing (old installs or custom packaging).  Never
    raises — callers should treat a missing cost model as "no routing,
    fall back to whatever threshold was before".
    """
    if not _DEFAULT_BASELINE_PATH.exists():
        return None
    try:
        return BackendCostModel.from_json(
            _DEFAULT_BASELINE_PATH,
            default_rank_count=default_rank_count,
            **kwargs,
        )
    except Exception:
        return None
