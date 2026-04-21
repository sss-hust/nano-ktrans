"""
Expert Map Store (fMoE, arXiv:2502.05370, EuroSys'26).

Records, for every past inference iteration, the full per-layer gate
probability distribution together with a prompt-level semantic embedding.
When a new request arrives we match it against the store in two phases:

1. **Semantic phase** (``layer_idx < prefetch_distance``):
   Compare the new prompt embedding against historical prompt embeddings
   via cosine similarity. Use the matched iteration's early-layer
   distributions to drive cold-start prefetching — we have no trajectory
   yet at this point.

2. **Trajectory phase** (``layer_idx >= prefetch_distance``):
   Compare the already-observed gate distributions of the current
   iteration against the historical trajectories. Use the best match to
   drive prefetching for layers ``layer_idx + prefetch_distance ..``.

The goal is to steer :class:`ExpertMaterializationManager` prefetch
candidates towards the experts that historically co-occur with the
current prompt / trajectory, closing the "each decode step only sees the
current routing" blind spot of :mod:`scheduler.dynamic_expert_scheduler`.

Design notes
============

* **Hashable-free**: maps are ordinary Python objects; no pickling.
* **Bounded capacity** via LRU-like eviction; callers set ``capacity``.
* **Detached tensors**: stored distributions are ``torch.float32`` CPU
  tensors so they do not pin GPU memory or participate in autograd.
* **Thread-safe**: a single :class:`threading.RLock` guards all
  mutations so the background offload worker can safely search from
  another thread.
* **No learning**: follows fMoE's design decision to stay heuristic-only
  (see fMoE §7 — NN predictors are too expensive and don't transfer well
  across the fine-grained regime).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class ExpertMap:
    """One past iteration's routing fingerprint."""

    prompt_embedding: torch.Tensor  # [hidden_size], float32, L2-normalized
    layer_distributions: Dict[int, torch.Tensor] = field(default_factory=dict)
    # layer_idx -> [num_experts] float32 probabilities (already L2-normalized
    # for cosine similarity; raw probs kept separately for prefetch picking)
    layer_distributions_raw: Dict[int, torch.Tensor] = field(default_factory=dict)

    def record_layer(self, layer_idx: int, probs: torch.Tensor) -> None:
        raw = probs.detach().to(dtype=torch.float32, device="cpu").contiguous()
        # Guard against all-zero rows (e.g. dropped layers) — cosine similarity
        # on a zero vector is undefined; we store a normalized zero to keep
        # downstream code branch-free.
        norm = torch.linalg.norm(raw)
        if norm > 0:
            unit = raw / norm
        else:
            unit = raw
        self.layer_distributions_raw[int(layer_idx)] = raw
        self.layer_distributions[int(layer_idx)] = unit


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


class ExpertMapStore:
    """
    Bounded-capacity store of :class:`ExpertMap` fingerprints.

    The store is **append-only within a request**: callers build a single
    in-flight :class:`ExpertMap` via :meth:`begin_iteration`, fill its
    per-layer distributions during forward, then commit it with
    :meth:`commit_iteration`. This mirrors fMoE's pub-sub design where
    the Expert Map Store is updated only at well-defined iteration
    boundaries.

    Thread safety: all public methods acquire an internal ``RLock``. The
    in-flight map is per-request and not protected — callers must not
    share a single in-flight handle across threads.
    """

    def __init__(
        self,
        *,
        capacity: int = 1024,
        prefetch_distance: int = 3,
        semantic_fallback_score: float = 0.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.prefetch_distance = max(1, int(prefetch_distance))
        self.semantic_fallback_score = float(semantic_fallback_score)

        # OrderedDict with insertion == LRU order; recent iterations go to
        # the end, oldest popped from the front on overflow.
        self._maps: "OrderedDict[int, ExpertMap]" = OrderedDict()
        self._next_id = 0
        self._lock = RLock()

        # Diagnostics
        self.commit_count = 0
        self.eviction_count = 0
        self.semantic_queries = 0
        self.trajectory_queries = 0
        self.semantic_hits = 0
        self.trajectory_hits = 0

    # ── In-flight map lifecycle ───────────────────────────────────────

    def begin_iteration(self, prompt_embedding: torch.Tensor) -> ExpertMap:
        """Create a new in-flight map for the current iteration."""
        embedding = prompt_embedding.detach().to(dtype=torch.float32, device="cpu").contiguous()
        return ExpertMap(prompt_embedding=_l2_normalize(embedding))

    def commit_iteration(self, expert_map: ExpertMap) -> int:
        """Insert a fully-populated map. Returns the assigned map id."""
        with self._lock:
            map_id = self._next_id
            self._next_id += 1
            self._maps[map_id] = expert_map
            self.commit_count += 1
            while len(self._maps) > self.capacity:
                self._maps.popitem(last=False)
                self.eviction_count += 1
            return map_id

    # ── Search APIs ───────────────────────────────────────────────────

    def semantic_search(
        self,
        prompt_embedding: torch.Tensor,
        *,
        layer_idx: int,
        top_k: int,
    ) -> List[int]:
        """
        Cold-start prefetch search (layer_idx < prefetch_distance).

        Uses prompt semantic similarity only. Returns a list of up to
        ``top_k`` expert ids ordered by predicted activation probability
        in ``layer_idx``. Returns empty list if the store has no
        historical layer-``layer_idx`` distribution.
        """
        if top_k <= 0:
            return []
        with self._lock:
            self.semantic_queries += 1
            if not self._maps:
                return []
            query = _l2_normalize(
                prompt_embedding.detach().to(dtype=torch.float32, device="cpu").contiguous()
            )
            best_score = self.semantic_fallback_score
            best_probs: Optional[torch.Tensor] = None
            for candidate in self._maps.values():
                dist = candidate.layer_distributions_raw.get(int(layer_idx))
                if dist is None:
                    continue
                score = float(torch.dot(query, candidate.prompt_embedding).item())
                if score > best_score:
                    best_score = score
                    best_probs = dist
            if best_probs is None:
                return []
            self.semantic_hits += 1
            return self._pick_top_experts(best_probs, top_k=top_k)

    def trajectory_search(
        self,
        *,
        observed: Dict[int, torch.Tensor],
        target_layer_idx: int,
        top_k: int,
    ) -> List[int]:
        """
        Warm-path prefetch search (target_layer_idx >= prefetch_distance).

        ``observed`` maps already-seen layer indices to their raw gate
        probability distributions in the current iteration. The match
        score is the averaged cosine similarity across shared layers.
        """
        if top_k <= 0 or not observed:
            return []
        with self._lock:
            self.trajectory_queries += 1
            if not self._maps:
                return []

            observed_units: Dict[int, torch.Tensor] = {}
            for layer_idx, probs in observed.items():
                vec = probs.detach().to(dtype=torch.float32, device="cpu").contiguous()
                unit = _l2_normalize(vec)
                observed_units[int(layer_idx)] = unit

            best_score = self.semantic_fallback_score
            best_probs: Optional[torch.Tensor] = None
            for candidate in self._maps.values():
                target_dist = candidate.layer_distributions_raw.get(int(target_layer_idx))
                if target_dist is None:
                    continue
                score_sum = 0.0
                matched = 0
                for layer_idx, obs_unit in observed_units.items():
                    hist_unit = candidate.layer_distributions.get(layer_idx)
                    if hist_unit is None:
                        continue
                    score_sum += float(torch.dot(obs_unit, hist_unit).item())
                    matched += 1
                if matched == 0:
                    continue
                avg_score = score_sum / matched
                if avg_score > best_score:
                    best_score = avg_score
                    best_probs = target_dist
            if best_probs is None:
                return []
            self.trajectory_hits += 1
            return self._pick_top_experts(best_probs, top_k=top_k)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pick_top_experts(probs: torch.Tensor, *, top_k: int) -> List[int]:
        if probs.numel() == 0:
            return []
        k = min(int(top_k), int(probs.numel()))
        _, indices = torch.topk(probs, k=k)
        return [int(idx) for idx in indices.tolist()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._maps)

    def clear(self) -> None:
        with self._lock:
            self._maps.clear()
            self._next_id = 0

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "capacity": self.capacity,
                "prefetch_distance": self.prefetch_distance,
                "size": len(self._maps),
                "commit_count": self.commit_count,
                "eviction_count": self.eviction_count,
                "semantic_queries": self.semantic_queries,
                "semantic_hits": self.semantic_hits,
                "trajectory_queries": self.trajectory_queries,
                "trajectory_hits": self.trajectory_hits,
            }
