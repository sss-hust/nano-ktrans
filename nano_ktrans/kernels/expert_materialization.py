from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from queue import Empty, Queue
from threading import Event, Lock, Thread
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
        self._ready_queue = deque()
        self._ready_mark_queue: "Queue[ExpertKey | None]" = Queue()
        self._resolve_queue: "Queue[ExpertKey | None]" = Queue()
        self._stop_event = Event()
        self._lock = Lock()
        self._resolver_thread: Optional[Thread] = None
        self._ready_callback = None
        if self.executor is not None:
            self._resolver_thread = Thread(
                target=self._resolve_ready_loop,
                name="expert-prefetch-resolver",
                daemon=True,
            )
            self._resolver_thread.start()

        self.prefetch_submitted = 0
        self.prefetch_resolved = 0
        self.prefetch_polled_ready = 0
        self.prefetch_completion_events = 0
        self.prefetch_background_resolved = 0
        self.prefetch_background_failures = 0
        self.prefetch_background_ready_callbacks = 0
        self.resident_stage_hits = 0
        self.cache_hits = 0
        self.sync_loads = 0
        self.cache_evictions = 0

    def _cache_key(self, layer_idx: int, expert_idx: int) -> ExpertKey:
        return (int(layer_idx), int(expert_idx))

    def has_cached(self, layer_idx: int, expert_idx: int) -> bool:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            return key in self._cache

    def has_pending_or_ready(self) -> bool:
        with self._lock:
            return bool(self._futures) or bool(self._ready_queue)

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

    def set_ready_callback(self, callback) -> None:
        self._ready_callback = callback

    def _store_cache(self, key: ExpertKey, weights: ExpertWeights) -> None:
        self._cache[key] = weights
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached_experts:
            self._cache.popitem(last=False)
            self.cache_evictions += 1

    def stage_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        weights: ExpertWeights,
    ) -> bool:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return False
            self._store_cache(key, {name: tensor.contiguous().cpu() for name, tensor in weights.items()})
            self.prefetch_resolved += 1
            self.resident_stage_hits += 1
        return True

    def _on_prefetch_done(self, key: ExpertKey, future: Future) -> None:
        with self._lock:
            tracked = self._futures.get(key)
            if tracked is not future:
                return
            self.prefetch_completion_events += 1
        if self._resolver_thread is None:
            with self._lock:
                self._ready_queue.append(key)
            return
        self._resolve_queue.put(key)

    def _resolve_completed_future(self, key: ExpertKey) -> bool:
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            return False

        try:
            weights = future.result()
        except Exception:
            with self._lock:
                tracked = self._futures.get(key)
                if tracked is future:
                    self._futures.pop(key, None)
                    self.prefetch_background_failures += 1
            return False

        with self._lock:
            tracked = self._futures.get(key)
            if tracked is not future:
                return False
            self._futures.pop(key, None)
            self._store_cache(key, weights)
            self.prefetch_resolved += 1
            self._ready_queue.append(key)
            self._ready_mark_queue.put(key)
        return True

    def _resolve_ready_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                key = self._resolve_queue.get(timeout=0.05)
            except Empty:
                continue
            if key is None:
                self._resolve_queue.task_done()
                break
            resolved = self._resolve_completed_future(key)
            with self._lock:
                if resolved:
                    self.prefetch_background_resolved += 1
            self._resolve_queue.task_done()

    def drain_ready_callbacks(self) -> int:
        if self._ready_callback is None:
            return 0
        callbacks = 0
        while True:
            try:
                key = self._ready_mark_queue.get_nowait()
            except Empty:
                break
            if key is None:
                self._ready_mark_queue.task_done()
                continue
            layer_idx, expert_idx = key
            self._ready_callback(int(layer_idx), int(expert_idx))
            callbacks += 1
            self._ready_mark_queue.task_done()
        with self._lock:
            self.prefetch_background_ready_callbacks += callbacks
        return callbacks

    def prefetch(self, layer_idx: int, expert_idx: int) -> bool:
        key = self._cache_key(layer_idx, expert_idx)
        with self._lock:
            if key in self._cache or key in self._futures:
                return False
            self.prefetch_submitted += 1
            if self.executor is None:
                weights = self._load_expert(layer_idx, expert_idx)
                self._store_cache(key, weights)
                self.prefetch_resolved += 1
                return True
            future = self.executor.submit(self._load_expert, layer_idx, expert_idx)
            self._futures[key] = future
            future.add_done_callback(lambda done_future, ready_key=key: self._on_prefetch_done(ready_key, done_future))
            return True

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
            self._resolve_completed_future(key)
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
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

    def poll_ready(self) -> list[ExpertKey]:
        ready_keys: list[ExpertKey] = []
        with self._lock:
            while self._ready_queue:
                ready_keys.append(self._ready_queue.popleft())
            self.prefetch_polled_ready += len(ready_keys)
        return ready_keys

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._resolver_thread is not None:
            self._resolve_queue.put(None)
            self._resolver_thread.join(timeout=1.0)
        self._ready_mark_queue.put(None)
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "pending_prefetches": len(self._futures),
                "max_cached_experts": self.max_cached_experts,
                "prefetch_workers": self.prefetch_workers,
                "background_resolver_enabled": self._resolver_thread is not None,
                "prefetch_submitted": self.prefetch_submitted,
                "prefetch_resolved": self.prefetch_resolved,
                "prefetch_polled_ready": self.prefetch_polled_ready,
                "prefetch_completion_events": self.prefetch_completion_events,
                "prefetch_background_resolved": self.prefetch_background_resolved,
                "prefetch_background_failures": self.prefetch_background_failures,
                "prefetch_background_ready_callbacks": self.prefetch_background_ready_callbacks,
                "resident_stage_hits": self.resident_stage_hits,
                "cache_hits": self.cache_hits,
                "sync_loads": self.sync_loads,
                "cache_evictions": self.cache_evictions,
                "cached_keys": [
                    {"layer_idx": layer_idx, "expert_idx": expert_idx}
                    for (layer_idx, expert_idx) in self._cache.keys()
                ],
            }
