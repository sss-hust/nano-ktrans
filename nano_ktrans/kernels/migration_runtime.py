from __future__ import annotations

from typing import Any, Iterable

import torch


class MigrationPipelineRuntime:
    """
    Token-step 级的迁移流水线运行时。

    当前目标不是直接引入真正的后台线程或 GPU<->PIM DMA，而是先把：
    - ready prefetch 的轮询
    - ready promotion 的预应用
    - token-step 级诊断
    收敛到一个更接近独立 runtime hook 的地方。
    """

    def __init__(self) -> None:
        self.tick_calls = 0
        self.ready_polled_total = 0
        self.ready_applied_total = 0
        self.ready_deferred_total = 0
        self.layers_touched_total = 0
        self.last_phase = ""

    def tick_layers(self, decoder_layers: Iterable[Any], *, phase: str) -> dict[str, int | str]:
        ready_polled = 0
        ready_applied = 0
        ready_deferred = 0
        layers_touched = 0

        for decoder_layer in decoder_layers:
            hybrid_moe = getattr(decoder_layer, "hybrid_moe", None)
            if hybrid_moe is None:
                continue
            layers_touched += 1

            advance_fn = getattr(hybrid_moe, "advance_offload_pipeline", None)
            if advance_fn is None:
                refresh_fn = getattr(hybrid_moe, "refresh_offload_state", None)
                if refresh_fn is None:
                    continue
                ready_polled += int(refresh_fn())
                continue

            first_param = next(iter(decoder_layer.parameters()), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
            dtype = first_param.dtype if first_param is not None else torch.float32
            stats = advance_fn(phase=phase, device=device, dtype=dtype)
            ready_polled += int(stats.get("ready_polled", 0))
            ready_applied += int(stats.get("ready_applied", 0))
            ready_deferred += int(stats.get("ready_deferred", 0))

        self.tick_calls += 1
        self.ready_polled_total += ready_polled
        self.ready_applied_total += ready_applied
        self.ready_deferred_total += ready_deferred
        self.layers_touched_total += layers_touched
        self.last_phase = phase

        return {
            "phase": phase,
            "ready_polled": ready_polled,
            "ready_applied": ready_applied,
            "ready_deferred": ready_deferred,
            "layers_touched": layers_touched,
        }

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "offload_refresh_calls": int(self.tick_calls),
            "offload_refresh_ready_total": int(self.ready_polled_total),
            "offload_pipeline_ticks": int(self.tick_calls),
            "offload_pipeline_ready_applied_total": int(self.ready_applied_total),
            "offload_pipeline_ready_deferred_total": int(self.ready_deferred_total),
            "offload_pipeline_layers_touched_total": int(self.layers_touched_total),
            "offload_pipeline_last_phase": self.last_phase,
        }
