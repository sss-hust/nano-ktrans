from __future__ import annotations

from typing import Any


def summarize_offload_diagnostics(offload_diagnostics: dict[str, Any]) -> dict[str, Any]:
    scheduler = offload_diagnostics.get("dynamic_scheduler", {})
    layers = offload_diagnostics.get("layers", [])

    summary = {
        "profile": offload_diagnostics.get("scheduler_profile", {}),
        "enabled": bool(scheduler.get("enabled", False)),
        "layer_count": int(offload_diagnostics.get("layer_count", 0)),
        "prefetch_requested": 0,
        "prefetch_enqueued": 0,
        "prefetch_materialized": 0,
        "prefetch_candidate_scans": 0,
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
        "migration_deferred_events": 0,
        "migration_applied_events": 0,
        "migration_lifecycle_counts": {
            "queued": 0,
            "prefetching": 0,
            "ready": 0,
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
            summary["migration_pending_ops"] += int(migration_layer.get("pending_ops", 0))
            lifecycle_counts = migration_layer.get("lifecycle_state_counts", {})
            if migration_layer.get("pending_ops", 0) > 0 and lifecycle_counts.get("ready", 0) > 0:
                summary["migration_ready_only_drains"] += 1
            summary["migration_prefetching_events"] += int(
                migration_layer.get("total_prefetching_events", 0)
            )
            summary["migration_ready_events"] += int(migration_layer.get("total_ready_events", 0))
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
    return summary
