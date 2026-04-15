from __future__ import annotations

from typing import Any


PROFILE_SWEEP_SORT_KEYS = (
    "decode_tokens_per_second",
    "pipeline_prefetch_overlap_hits",
    "pipeline_promotion_source_activated",
    "pipeline_promotion_source_warm",
    "pipeline_promotion_source_cold",
    "pipeline_apply_batch_size_avg",
    "pipeline_apply_batch_evictions",
    "runtime_deferred_for_prefetch",
)


def summarize_offload_diagnostics(offload_diagnostics: dict[str, Any]) -> dict[str, Any]:
    scheduler = offload_diagnostics.get("dynamic_scheduler", {})
    layers = offload_diagnostics.get("layers", [])

    summary = {
        "profile": offload_diagnostics.get("scheduler_profile", {}),
        "enabled": bool(scheduler.get("enabled", False)),
        "layer_count": int(offload_diagnostics.get("layer_count", 0)),
        "offload_refresh_calls": int((offload_diagnostics.get("offload_refresh") or {}).get("offload_refresh_calls", 0)),
        "offload_refresh_ready_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_refresh_ready_total", 0)
        ),
        "offload_pipeline_ticks": int((offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_ticks", 0)),
        "offload_pipeline_prefetch_submitted_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_prefetch_submitted_total", 0)
        ),
        "offload_pipeline_activation_ready_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_activation_ready_total", 0)
        ),
        "offload_pipeline_ready_applied_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_ready_applied_total", 0)
        ),
        "offload_pipeline_ready_deferred_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_ready_deferred_total", 0)
        ),
        "offload_pipeline_layers_touched_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_layers_touched_total", 0)
        ),
        "prefetch_requested": 0,
        "prefetch_enqueued": 0,
        "prefetch_materialized": 0,
        "prefetch_candidate_scans": 0,
        "prefetch_polled_ready": 0,
        "prefetch_completion_events": 0,
        "pipeline_ticks": 0,
        "pipeline_ready_applied": 0,
        "pipeline_ready_deferred": 0,
        "pipeline_prefetch_overlap_hits": 0,
        "pipeline_promotion_source_activated": 0,
        "pipeline_promotion_source_warm": 0,
        "pipeline_promotion_source_cold": 0,
        "pipeline_apply_batches": 0,
        "pipeline_apply_batch_experts": 0,
        "pipeline_apply_batch_evictions": 0,
        "activation_submitted": 0,
        "activation_ready": 0,
        "activation_applied": 0,
        "activated_cache_hits": 0,
        "activated_cache_stores": 0,
        "activated_cache_evictions": 0,
        "activated_cache_size": 0,
        "decode_prefetch_hits": 0,
        "decode_prefetch_misses": 0,
        "runtime_evictions": 0,
        "runtime_deferred_for_prefetch": 0,
        "runtime_skipped_demotion_cooldown": 0,
        "applied_migration_ops": 0,
        "layers_with_pending_migrations": 0,
        "migration_submit_calls": 0,
        "migration_total_enqueued_ops": 0,
        "migration_total_deduped_ops": 0,
        "migration_total_drained_ops": 0,
        "migration_ready_only_drains": 0,
        "migration_pending_ops": 0,
        "migration_prefetching_events": 0,
        "migration_ready_events": 0,
        "migration_requeue_preserved_states": 0,
        "migration_stage_skips": 0,
        "migration_deferred_events": 0,
        "migration_applied_events": 0,
        "migration_lifecycle_counts": {
            "queued": 0,
            "prefetching": 0,
            "ready": 0,
            "warmed": 0,
            "activated": 0,
            "deferred": 0,
            "applied": 0,
        },
        "prefetch_hit_rate": None,
        "dedupe_ratio": None,
        "decode_ready_rate": None,
    }

    for layer in layers:
        summary["prefetch_requested"] += int(layer.get("prefetch_requested", 0))
        summary["prefetch_enqueued"] += int(layer.get("prefetch_enqueued", 0))
        summary["prefetch_materialized"] += int(layer.get("prefetch_materialized", 0))
        summary["prefetch_candidate_scans"] += int(layer.get("prefetch_candidate_scans", 0))
        summary["prefetch_polled_ready"] += int(
            (layer.get("materialization_manager") or {}).get("prefetch_polled_ready", 0)
        )
        summary["prefetch_completion_events"] += int(
            (layer.get("materialization_manager") or {}).get("prefetch_completion_events", 0)
        )
        summary["pipeline_ticks"] += int(layer.get("pipeline_ticks", 0))
        summary["pipeline_ready_applied"] += int(layer.get("pipeline_ready_applied", 0))
        summary["pipeline_ready_deferred"] += int(layer.get("pipeline_ready_deferred", 0))
        summary["pipeline_prefetch_overlap_hits"] += int(layer.get("pipeline_prefetch_overlap_hits", 0))
        summary["pipeline_promotion_source_activated"] += int(
            layer.get("pipeline_promotion_source_activated", 0)
        )
        summary["pipeline_promotion_source_warm"] += int(layer.get("pipeline_promotion_source_warm", 0))
        summary["pipeline_promotion_source_cold"] += int(layer.get("pipeline_promotion_source_cold", 0))
        summary["pipeline_apply_batches"] += int(layer.get("pipeline_apply_batches", 0))
        summary["pipeline_apply_batch_experts"] += int(layer.get("pipeline_apply_batch_experts", 0))
        summary["pipeline_apply_batch_evictions"] += int(layer.get("pipeline_apply_batch_evictions", 0))
        summary["activation_submitted"] += int(layer.get("activation_submitted", 0))
        summary["activation_ready"] += int(layer.get("activation_ready", 0))
        summary["activation_applied"] += int(layer.get("activation_applied", 0))
        summary["activated_cache_hits"] += int(layer.get("activated_cache_hits", 0))
        summary["activated_cache_stores"] += int(layer.get("activated_cache_stores", 0))
        summary["activated_cache_evictions"] += int(layer.get("activated_cache_evictions", 0))
        summary["activated_cache_size"] += int(layer.get("activated_cache_size", 0))
        summary["decode_prefetch_hits"] += int(layer.get("decode_prefetch_hits", 0))
        summary["decode_prefetch_misses"] += int(layer.get("decode_prefetch_misses", 0))
        summary["runtime_evictions"] += int(layer.get("runtime_evictions", 0))
        summary["runtime_deferred_for_prefetch"] += int(layer.get("runtime_deferred_for_prefetch", 0))
        summary["runtime_skipped_demotion_cooldown"] += int(
            layer.get("runtime_skipped_demotion_cooldown", 0)
        )
        summary["applied_migration_ops"] += int(layer.get("applied_migration_ops", 0))
        if layer.get("pending_migrations"):
            summary["layers_with_pending_migrations"] += 1

        backend = layer.get("backend") or {}
        summary["migration_submit_calls"] += int(backend.get("migration_submit_calls", 0))
        for migration_layer in backend.get("migration_manager", {}).get("layers", []):
            summary["migration_total_enqueued_ops"] += int(migration_layer.get("total_enqueued_ops", 0))
            summary["migration_total_deduped_ops"] += int(migration_layer.get("total_deduped_ops", 0))
            summary["migration_total_drained_ops"] += int(migration_layer.get("total_drained_ops", 0))
            summary["migration_ready_only_drains"] += int(migration_layer.get("total_ready_drains", 0))
            summary["migration_pending_ops"] += int(migration_layer.get("pending_ops", 0))
            lifecycle_counts = migration_layer.get("lifecycle_state_counts", {})
            summary["migration_prefetching_events"] += int(
                migration_layer.get("total_prefetching_events", 0)
            )
            summary["migration_ready_events"] += int(migration_layer.get("total_ready_events", 0))
            summary.setdefault("migration_warmed_events", 0)
            summary["migration_warmed_events"] += int(migration_layer.get("total_warmed_events", 0))
            summary.setdefault("migration_activated_events", 0)
            summary["migration_activated_events"] += int(migration_layer.get("total_activated_events", 0))
            summary["migration_requeue_preserved_states"] += int(
                migration_layer.get("total_requeue_preserved_states", 0)
            )
            summary["migration_stage_skips"] += int(migration_layer.get("total_stage_skips", 0))
            summary["migration_deferred_events"] += int(migration_layer.get("total_deferred_events", 0))
            summary["migration_applied_events"] += int(migration_layer.get("total_applied_events", 0))
            for key in summary["migration_lifecycle_counts"]:
                summary["migration_lifecycle_counts"][key] += int(lifecycle_counts.get(key, 0))

    total_prefetch_decisions = summary["decode_prefetch_hits"] + summary["decode_prefetch_misses"]
    if total_prefetch_decisions > 0:
        ready_rate = summary["decode_prefetch_hits"] / total_prefetch_decisions
        summary["prefetch_hit_rate"] = ready_rate
        summary["decode_ready_rate"] = ready_rate
    if summary["migration_total_enqueued_ops"] > 0:
        summary["dedupe_ratio"] = (
            summary["migration_total_deduped_ops"] / summary["migration_total_enqueued_ops"]
        )
    if summary["pipeline_apply_batches"] > 0:
        summary["pipeline_apply_batch_size_avg"] = (
            summary["pipeline_apply_batch_experts"] / summary["pipeline_apply_batches"]
        )
    else:
        summary["pipeline_apply_batch_size_avg"] = None
    return summary


def summarize_profile_sweep_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    best_profile: dict[str, Any] | None = None
    best_decode_tps = float("-inf")

    for result in results:
        if result.get("status") != "ok":
            continue
        scheduler_summary = result.get("scheduler_summary") or {}
        runs = result.get("runs") or []
        if not runs:
            continue

        decode_tps_values = [
            float(run["decode_tokens_per_second"])
            for run in runs
            if run.get("decode_tokens_per_second") is not None
        ]
        decode_tps_avg = (
            sum(decode_tps_values) / len(decode_tps_values)
            if decode_tps_values
            else None
        )

        item = {
            "backend": result.get("backend"),
            "scheduler_profile": result.get("scheduler_profile"),
            "decode_tokens_per_second": decode_tps_avg,
            "prefill_seconds_avg": (
                sum(float(run["prefill_seconds"]) for run in runs) / len(runs)
                if runs
                else None
            ),
            "decode_seconds_avg": (
                sum(float(run["decode_seconds"]) for run in runs) / len(runs)
                if runs
                else None
            ),
            "pipeline_prefetch_overlap_hits": int(
                scheduler_summary.get("pipeline_prefetch_overlap_hits", 0)
            ),
            "pipeline_promotion_source_activated": int(
                scheduler_summary.get("pipeline_promotion_source_activated", 0)
            ),
            "pipeline_promotion_source_warm": int(
                scheduler_summary.get("pipeline_promotion_source_warm", 0)
            ),
            "pipeline_promotion_source_cold": int(
                scheduler_summary.get("pipeline_promotion_source_cold", 0)
            ),
            "pipeline_apply_batches": int(
                scheduler_summary.get("pipeline_apply_batches", 0)
            ),
            "pipeline_apply_batch_size_avg": scheduler_summary.get(
                "pipeline_apply_batch_size_avg"
            ),
            "pipeline_apply_batch_evictions": int(
                scheduler_summary.get("pipeline_apply_batch_evictions", 0)
            ),
            "runtime_deferred_for_prefetch": int(
                scheduler_summary.get("runtime_deferred_for_prefetch", 0)
            ),
        }
        profiles.append(item)

        if decode_tps_avg is not None and decode_tps_avg > best_decode_tps:
            best_decode_tps = decode_tps_avg
            best_profile = item

    return {
        "sort_keys": list(PROFILE_SWEEP_SORT_KEYS),
        "profiles": profiles,
        "best_by_decode_tokens_per_second": best_profile,
    }
