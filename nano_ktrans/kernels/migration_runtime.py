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
        self.background_ticks = 0
        self.background_work_items_total = 0
        self.background_warm_prebuilt_total = 0
        self.background_activation_ready_total = 0
        self.background_activation_applied_total = 0
        self.background_apply_queue_enqueued_total = 0
        self.background_apply_commit_queue_enqueued_total = 0
        self.prefetch_submitted_total = 0
        self.ready_polled_total = 0
        self.activation_ready_total = 0
        self.ready_applied_total = 0
        self.ready_deferred_total = 0
        self.apply_batch_count_total = 0
        self.apply_batch_experts_total = 0
        self.apply_batch_evictions_total = 0
        self.apply_batch_activated_total = 0
        self.apply_batch_warm_total = 0
        self.apply_batch_cold_total = 0
        self.layers_touched_total = 0
        self.background_ready_callback_total = 0
        self.last_phase = ""

    def _tick_layers_impl(
        self,
        decoder_layers: Iterable[Any],
        *,
        phase: str,
        background_only: bool,
    ) -> dict[str, int | str]:
        ready_polled = 0
        activation_ready = 0
        activation_applied = 0
        ready_applied = 0
        ready_deferred = 0
        prefetch_submitted = 0
        apply_batch_count = 0
        apply_batch_experts = 0
        apply_batch_evictions = 0
        apply_batch_activated = 0
        apply_batch_warm = 0
        apply_batch_cold = 0
        layers_touched = 0
        background_ready_callbacks = 0
        warm_prebuilt = 0
        background_work_items = 0
        background_apply_queue_enqueued = 0
        background_apply_commit_queue_enqueued = 0

        for decoder_layer in decoder_layers:
            hybrid_moe = getattr(decoder_layer, "hybrid_moe", None)
            if hybrid_moe is None:
                continue
            layers_touched += 1

            if background_only:
                background_advance_fn = getattr(hybrid_moe, "background_advance_offload_pipeline", None)
                if background_advance_fn is not None:
                    first_param = next(iter(decoder_layer.parameters()), None)
                    device = first_param.device if first_param is not None else torch.device("cpu")
                    dtype = first_param.dtype if first_param is not None else torch.float32
                    background_stats = background_advance_fn(phase=phase, device=device, dtype=dtype)
                    background_ready_callbacks += int(background_stats.get("ready_polled", 0))
                    warm_prebuilt += int(background_stats.get("warm_prebuilt", 0))
                    activation_ready += int(background_stats.get("activation_ready", 0))
                    activation_applied += int(background_stats.get("activation_applied", 0))
                    background_apply_queue_enqueued += int(
                        background_stats.get("apply_queue_enqueued", 0)
                    )
                    background_apply_commit_queue_enqueued += int(
                        background_stats.get("apply_commit_queue_enqueued", 0)
                    )
                    background_work_items += (
                        int(background_stats.get("ready_polled", 0))
                        + int(background_stats.get("warm_prebuilt", 0))
                        + int(background_stats.get("activation_ready", 0))
                        + int(background_stats.get("activation_applied", 0))
                        + int(background_stats.get("apply_queue_enqueued", 0))
                        + int(background_stats.get("apply_commit_queue_enqueued", 0))
                    )
                    continue
                background_tick_fn = getattr(hybrid_moe, "background_tick_offload_state", None)
                if background_tick_fn is None:
                    continue
                work_items = int(background_tick_fn())
                background_ready_callbacks += work_items
                background_work_items += work_items
                continue

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
            activation_ready += int(stats.get("activation_ready", 0))
            ready_applied += int(stats.get("ready_applied", 0))
            ready_deferred += int(stats.get("ready_deferred", 0))
            prefetch_submitted += int(stats.get("prefetch_submitted", 0))
            apply_batch_count += int(stats.get("apply_batch_count", 0))
            apply_batch_experts += int(stats.get("apply_batch_experts", 0))
            apply_batch_evictions += int(stats.get("apply_batch_evictions", 0))
            apply_batch_activated += int(stats.get("apply_batch_activated", 0))
            apply_batch_warm += int(stats.get("apply_batch_warm", 0))
            apply_batch_cold += int(stats.get("apply_batch_cold", 0))

        return {
            "phase": phase,
            "prefetch_submitted": prefetch_submitted,
            "ready_polled": ready_polled,
            "activation_ready": activation_ready,
            "ready_applied": ready_applied,
            "ready_deferred": ready_deferred,
            "layers_touched": layers_touched,
            "apply_batch_count": apply_batch_count,
            "apply_batch_experts": apply_batch_experts,
            "apply_batch_evictions": apply_batch_evictions,
            "apply_batch_activated": apply_batch_activated,
            "apply_batch_warm": apply_batch_warm,
            "apply_batch_cold": apply_batch_cold,
            "background_ready_callbacks": background_ready_callbacks,
            "background_work_items": background_work_items,
            "background_warm_prebuilt": warm_prebuilt,
            "background_activation_ready": activation_ready,
            "background_activation_applied": activation_applied,
            "background_apply_queue_enqueued": background_apply_queue_enqueued,
            "background_apply_commit_queue_enqueued": background_apply_commit_queue_enqueued,
            "background_only": int(background_only),
        }

    def background_tick_layers(self, decoder_layers: Iterable[Any], *, phase: str) -> dict[str, int | str]:
        stats = self._tick_layers_impl(decoder_layers, phase=phase, background_only=True)
        self.background_ticks += 1
        self.background_ready_callback_total += int(stats.get("background_ready_callbacks", 0))
        self.background_work_items_total += int(stats.get("background_work_items", 0))
        self.background_warm_prebuilt_total += int(stats.get("background_warm_prebuilt", 0))
        self.background_activation_ready_total += int(stats.get("background_activation_ready", 0))
        self.background_activation_applied_total += int(stats.get("background_activation_applied", 0))
        self.background_apply_queue_enqueued_total += int(
            stats.get("background_apply_queue_enqueued", 0)
        )
        self.background_apply_commit_queue_enqueued_total += int(
            stats.get("background_apply_commit_queue_enqueued", 0)
        )
        self.layers_touched_total += int(stats.get("layers_touched", 0))
        self.last_phase = phase
        return stats

    def tick_layers(self, decoder_layers: Iterable[Any], *, phase: str) -> dict[str, int | str]:
        stats = self._tick_layers_impl(decoder_layers, phase=phase, background_only=False)
        ready_polled = int(stats.get("ready_polled", 0))
        activation_ready = int(stats.get("activation_ready", 0))
        ready_applied = int(stats.get("ready_applied", 0))
        ready_deferred = int(stats.get("ready_deferred", 0))
        prefetch_submitted = int(stats.get("prefetch_submitted", 0))
        apply_batch_count = int(stats.get("apply_batch_count", 0))
        apply_batch_experts = int(stats.get("apply_batch_experts", 0))
        apply_batch_evictions = int(stats.get("apply_batch_evictions", 0))
        apply_batch_activated = int(stats.get("apply_batch_activated", 0))
        apply_batch_warm = int(stats.get("apply_batch_warm", 0))
        apply_batch_cold = int(stats.get("apply_batch_cold", 0))
        layers_touched = int(stats.get("layers_touched", 0))
        self.tick_calls += 1
        self.prefetch_submitted_total += prefetch_submitted
        self.ready_polled_total += ready_polled
        self.activation_ready_total += activation_ready
        self.ready_applied_total += ready_applied
        self.ready_deferred_total += ready_deferred
        self.apply_batch_count_total += apply_batch_count
        self.apply_batch_experts_total += apply_batch_experts
        self.apply_batch_evictions_total += apply_batch_evictions
        self.apply_batch_activated_total += apply_batch_activated
        self.apply_batch_warm_total += apply_batch_warm
        self.apply_batch_cold_total += apply_batch_cold
        self.layers_touched_total += layers_touched
        self.last_phase = phase

        return stats

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "offload_refresh_calls": int(self.tick_calls),
            "offload_background_ticks": int(self.background_ticks),
            "offload_background_work_items_total": int(self.background_work_items_total),
            "offload_background_warm_prebuilt_total": int(self.background_warm_prebuilt_total),
            "offload_background_activation_ready_total": int(self.background_activation_ready_total),
            "offload_background_activation_applied_total": int(self.background_activation_applied_total),
            "offload_background_apply_queue_enqueued_total": int(self.background_apply_queue_enqueued_total),
            "offload_background_apply_commit_queue_enqueued_total": int(
                self.background_apply_commit_queue_enqueued_total
            ),
            "offload_refresh_ready_total": int(self.ready_polled_total),
            "offload_pipeline_ticks": int(self.tick_calls),
            "offload_pipeline_prefetch_submitted_total": int(self.prefetch_submitted_total),
            "offload_pipeline_activation_ready_total": int(self.activation_ready_total),
            "offload_pipeline_ready_applied_total": int(self.ready_applied_total),
            "offload_pipeline_ready_deferred_total": int(self.ready_deferred_total),
            "offload_pipeline_apply_batch_count_total": int(self.apply_batch_count_total),
            "offload_pipeline_apply_batch_experts_total": int(self.apply_batch_experts_total),
            "offload_pipeline_apply_batch_evictions_total": int(self.apply_batch_evictions_total),
            "offload_pipeline_apply_batch_activated_total": int(self.apply_batch_activated_total),
            "offload_pipeline_apply_batch_warm_total": int(self.apply_batch_warm_total),
            "offload_pipeline_apply_batch_cold_total": int(self.apply_batch_cold_total),
            "offload_pipeline_layers_touched_total": int(self.layers_touched_total),
            "offload_pipeline_background_ready_callback_total": int(self.background_ready_callback_total),
            "offload_pipeline_last_phase": self.last_phase,
        }
