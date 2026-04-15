from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Dict, Optional, Tuple

import torch

from .weight_loader import ExpertWeightLoader


ExpertKey = Tuple[int, int]
ExpertWeights = Dict[str, torch.Tensor]


class ExpertMaterializationManager:
    """
    单 expert 权重的预取与 CPU staging cache。

    当前目标不是直接完成 GPU<->PIM 异步搬运，而是先把 decode promotion
    的最重同步点从“现场扫 safetensors”变成“prefill 阶段预取到 CPU 缓存，
    decode 时直接 materialize”。
    """

    def __init__(
        self,
        *,
        weight_path: str,
        expert_key_template: str,
        expert_proj_names: Optional[Dict[str, str]] = None,
        max_cached_experts: int = 8,
        prefetch_workers: int = 1,
    ) -> None:
        self.loader = ExpertWeightLoader(weight_path)
        self.expert_key_template = expert_key_template
        self.expert_proj_names = expert_proj_names
        self.max_cached_experts = max(1, int(max_cached_experts))
        self.prefetch_workers = max(0, int(prefetch_workers))
        self.executor = (
            ThreadPoolExecutor(
                max_workers=self.prefetch_workers,
                thread_name_prefix="expert-prefetch",
            )
            if self.prefetch_workers > 0
            else None
        )
        self._cache: OrderedDict[ExpertKey, ExpertWeights] = OrderedDict()
        self._futures: Dict[ExpertKey, Future] = {}
        self._lock = Lock()

        self.prefetch_submitted = 0
        self.prefetch_resolved = 0
        self.cache_hits = 0
        self.sync_loads = 0
        self.cache_evictions = 0

    def _cache_key(self, layer_idx: int, expert_idx: int) -> ExpertKey:
        return (int(layer_idx), int(expert_idx))

    def has_cached(self, layer_idx: int, expert_idx: int) -> bool:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            return key in self._cache

    def is_ready(self, layer_idx: int, expert_idx: int) -> bool:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                return True
            future = self._futures.get(key)
            return future is not None and future.done()

    def _load_expert(self, layer_idx: int, expert_idx: int) -> ExpertWeights:
        return self.loader.load_expert(
            layer_idx,
            expert_idx,
            key_template=self.expert_key_template,
            proj_name_map=self.expert_proj_names,
        )

    def _store_cache(self, key: ExpertKey, weights: ExpertWeights) -> None:
        self._cache[key] = weights
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached_experts:
            self._cache.popitem(last=False)
            self.cache_evictions += 1

    def prefetch(self, layer_idx: int, expert_idx: int) -> None:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            if key in self._cache or key in self._futures:
                return
            self.prefetch_submitted += 1
            if self.executor is None:
                weights = self._load_expert(layer_idx, expert_idx)
                self._store_cache(key, weights)
                self.prefetch_resolved += 1
                return
            self._futures[key] = self.executor.submit(self._load_expert, layer_idx, expert_idx)

    def get_expert(self, layer_idx: int, expert_idx: int) -> ExpertWeights:
        key = self._cache_key(layer_idx, expert_idx)
        future: Optional[Future] = None
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return cached
            future = self._futures.get(key)

        if future is not None:
            weights = future.result()
            with self._lock:
                self._futures.pop(key, None)
                self._store_cache(key, weights)
            self.prefetch_resolved += 1
            return weights

        weights = self._load_expert(layer_idx, expert_idx)
        with self._lock:
            self._store_cache(key, weights)
        self.sync_loads += 1
        return weights

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "pending_prefetches": len(self._futures),
                "max_cached_experts": self.max_cached_experts,
                "prefetch_workers": self.prefetch_workers,
                "prefetch_submitted": self.prefetch_submitted,
                "prefetch_resolved": self.prefetch_resolved,
                "cache_hits": self.cache_hits,
                "sync_loads": self.sync_loads,
                "cache_evictions": self.cache_evictions,
                "cached_keys": [
                    {"layer_idx": layer_idx, "expert_idx": expert_idx}
                    for (layer_idx, expert_idx) in self._cache.keys()
                ],
            }
