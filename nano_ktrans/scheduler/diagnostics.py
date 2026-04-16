from __future__ import annotations

from typing import Any


PROFILE_SWEEP_SORT_KEYS = (
    "decode_tokens_per_second",
    "pipeline_promotion_non_cold_total",
    "pipeline_promotion_non_cold_ratio",
    "prepared_cache_utilization",
    "effective_prepared_cache_utilization",
    "cold_promotion_penalty_avg",
    "prepared_cache_rebalance_pressure_avg",
    "pipeline_prefetch_overlap_hits",
    "pipeline_promotion_source_activated",
    "pipeline_promotion_source_warm",
    "pipeline_promotion_source_cold",
    "runtime_offload_pipeline_apply_batch_count_total",
    "runtime_offload_pipeline_apply_batch_experts_total",
    "runtime_offload_pipeline_apply_batch_evictions_total",
    "pipeline_apply_batch_size_avg",
    "pipeline_apply_batch_evictions",
    "migration_activation_eviction_regressions",
    "migration_warm_eviction_regressions",
    "runtime_deferred_for_prefetch",
)

PROFILE_SWEEP_METRIC_DIRECTIONS = {
    "decode_tokens_per_second": "max",
    "prefill_seconds_avg": "min",
    "decode_seconds_avg": "min",
    "pipeline_prefetch_overlap_hits": "max",
    "pipeline_promotion_source_activated": "max",
    "pipeline_promotion_source_warm": "max",
    "pipeline_promotion_source_cold": "min",
    "pipeline_promotion_non_cold_total": "max",
    "pipeline_promotion_non_cold_ratio": "max",
    "pipeline_apply_batches": "max",
    "pipeline_apply_batch_size_avg": "max",
    "pipeline_apply_batch_evictions": "min",
    "runtime_offload_pipeline_apply_batch_count_total": "max",
    "runtime_offload_pipeline_apply_batch_experts_total": "max",
    "runtime_offload_pipeline_apply_batch_evictions_total": "min",
    "runtime_apply_batch_size_avg": "max",
    "prepared_cache_utilization": "max",
    "effective_prepared_cache_utilization": "max",
    "cold_promotion_penalty_avg": "min",
    "prepared_cache_rebalance_pressure_avg": "min",
    "migration_activation_eviction_regressions": "min",
    "migration_warm_eviction_regressions": "min",
    "runtime_deferred_for_prefetch": "min",
}


def _metric_sort_key(value: Any, direction: str) -> tuple[int, float]:
    if value is None:
        return (0, 0.0)
    numeric = float(value)
    return (1, numeric if direction == "max" else -numeric)


def _summarize_best_profiles_by_metric(profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_metric: dict[str, dict[str, Any]] = {}
    for metric, direction in PROFILE_SWEEP_METRIC_DIRECTIONS.items():
        candidates = [profile for profile in profiles if profile.get(metric) is not None]
        if not candidates:
            continue
        best_profile = max(candidates, key=lambda profile: _metric_sort_key(profile.get(metric), direction))
        best_by_metric[metric] = {
            "backend": best_profile.get("backend"),
            "scheduler_profile": best_profile.get("scheduler_profile"),
            "value": best_profile.get(metric),
            "direction": direction,
        }
    return best_by_metric


def _build_profile_comparison_table(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        profiles,
        key=lambda profile: (
            float(profile.get("decode_tokens_per_second") or float("-inf")),
            float(profile.get("pipeline_promotion_non_cold_ratio") or float("-inf")),
            -float(profile.get("migration_activation_eviction_regressions") or 0.0),
            -float(profile.get("migration_warm_eviction_regressions") or 0.0),
            -float(profile.get("runtime_deferred_for_prefetch") or 0.0),
        ),
        reverse=True,
    )

    comparison_table: list[dict[str, Any]] = []
    for rank, profile in enumerate(ordered, start=1):
        row = dict(profile)
        row["rank_by_decode_tokens_per_second"] = rank
        row["row_id"] = f"{profile.get('backend')}:{profile.get('scheduler_profile')}"
        comparison_table.append(row)
    return comparison_table


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
        "offload_pipeline_apply_batch_count_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_apply_batch_count_total", 0)
        ),
        "offload_pipeline_apply_batch_experts_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_apply_batch_experts_total", 0)
        ),
        "offload_pipeline_apply_batch_evictions_total": int(
            (offload_diagnostics.get("offload_refresh") or {}).get("offload_pipeline_apply_batch_evictions_total", 0)
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
        "pipeline_apply_batch_activated": 0,
        "pipeline_apply_batch_warm": 0,
        "pipeline_apply_batch_cold": 0,
        "activation_submitted": 0,
        "activation_ready": 0,
        "activation_applied": 0,
        "activated_cache_hits": 0,
        "activated_cache_stores": 0,
        "activated_cache_evictions": 0,
        "activated_cache_size": 0,
        "prepared_cache_limit": 0,
        "effective_prepared_cache_limit": 0,
        "prepared_cache_size": 0,
        "effective_warm_cache_limit": 0,
        "prepared_cache_rebalance_pressure": 0.0,
        "prepared_cache_rebalance_evicted_warm": 0,
        "prepared_cache_rebalance_evicted_activated": 0,
        "prepared_cache_rebalance_demoted_to_warm": 0,
        "prepared_cache_rebalance_dropped_to_ready": 0,
        "prepared_cache_activation_stage_bonus": 0.0,
        "cold_promotion_penalty": 0.0,
        "adaptive_activation_limit": 0,
        "adaptive_prebuild_limit": 0,
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
        "migration_warm_eviction_regressions": 0,
        "migration_activation_eviction_regressions": 0,
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
        summary["pipeline_apply_batch_activated"] += int(layer.get("pipeline_apply_batch_activated", 0))
        summary["pipeline_apply_batch_warm"] += int(layer.get("pipeline_apply_batch_warm", 0))
        summary["pipeline_apply_batch_cold"] += int(layer.get("pipeline_apply_batch_cold", 0))
        summary["activation_submitted"] += int(layer.get("activation_submitted", 0))
        summary["activation_ready"] += int(layer.get("activation_ready", 0))
        summary["activation_applied"] += int(layer.get("activation_applied", 0))
        summary["activated_cache_hits"] += int(layer.get("activated_cache_hits", 0))
        summary["activated_cache_stores"] += int(layer.get("activated_cache_stores", 0))
        summary["activated_cache_evictions"] += int(layer.get("activated_cache_evictions", 0))
        summary["activated_cache_size"] += int(layer.get("activated_cache_size", 0))
        summary["prepared_cache_limit"] += int(layer.get("prepared_cache_limit") or 0)
        summary["effective_prepared_cache_limit"] += int(layer.get("effective_prepared_cache_limit") or 0)
        summary["prepared_cache_size"] += int(layer.get("prepared_cache_size", 0))
        summary["effective_warm_cache_limit"] += int(layer.get("effective_warm_cache_limit", 0))
        summary["prepared_cache_rebalance_pressure"] += float(
            layer.get("prepared_cache_rebalance_pressure", 0.0)
        )
        summary["prepared_cache_rebalance_evicted_warm"] += int(
            layer.get("prepared_cache_rebalance_evicted_warm", 0)
        )
        summary["prepared_cache_rebalance_evicted_activated"] += int(
            layer.get("prepared_cache_rebalance_evicted_activated", 0)
        )
        summary["prepared_cache_rebalance_demoted_to_warm"] += int(
            layer.get("prepared_cache_rebalance_demoted_to_warm", 0)
        )
        summary["prepared_cache_rebalance_dropped_to_ready"] += int(
            layer.get("prepared_cache_rebalance_dropped_to_ready", 0)
        )
        summary["prepared_cache_activation_stage_bonus"] += float(
            layer.get("prepared_cache_activation_stage_bonus", 0.0)
        )
        summary["cold_promotion_penalty"] += float(layer.get("cold_promotion_penalty", 0.0))
        summary["adaptive_activation_limit"] += int(layer.get("adaptive_activation_limit", 0))
        summary["adaptive_prebuild_limit"] += int(layer.get("adaptive_prebuild_limit", 0))
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
            summary["migration_warm_eviction_regressions"] += int(
                migration_layer.get("total_warm_eviction_regressions", 0)
            )
            summary["migration_activation_eviction_regressions"] += int(
                migration_layer.get("total_activation_eviction_regressions", 0)
            )
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
    if summary["prepared_cache_limit"] > 0:
        summary["prepared_cache_utilization"] = (
            summary["prepared_cache_size"] / summary["prepared_cache_limit"]
        )
    else:
        summary["prepared_cache_utilization"] = None
    if summary["effective_prepared_cache_limit"] > 0:
        summary["effective_prepared_cache_utilization"] = (
            summary["prepared_cache_size"] / summary["effective_prepared_cache_limit"]
        )
    else:
        summary["effective_prepared_cache_utilization"] = None
    if summary["layer_count"] > 0:
        summary["prepared_cache_activation_stage_bonus_avg"] = (
            summary["prepared_cache_activation_stage_bonus"] / summary["layer_count"]
        )
        summary["cold_promotion_penalty_avg"] = (
            summary["cold_promotion_penalty"] / summary["layer_count"]
        )
        summary["prepared_cache_rebalance_pressure_avg"] = (
            summary["prepared_cache_rebalance_pressure"] / summary["layer_count"]
        )
        summary["adaptive_activation_limit_avg"] = (
            summary["adaptive_activation_limit"] / summary["layer_count"]
        )
        summary["adaptive_prebuild_limit_avg"] = (
            summary["adaptive_prebuild_limit"] / summary["layer_count"]
        )
    else:
        summary["prepared_cache_activation_stage_bonus_avg"] = None
        summary["cold_promotion_penalty_avg"] = None
        summary["prepared_cache_rebalance_pressure_avg"] = None
        summary["adaptive_activation_limit_avg"] = None
        summary["adaptive_prebuild_limit_avg"] = None
    total_rebalance_events = (
        summary["prepared_cache_rebalance_evicted_warm"]
        + summary["prepared_cache_rebalance_evicted_activated"]
    )
    if total_rebalance_events > 0:
        summary["prepared_cache_rebalance_activated_ratio"] = (
            summary["prepared_cache_rebalance_evicted_activated"] / total_rebalance_events
        )
    else:
        summary["prepared_cache_rebalance_activated_ratio"] = None
    if summary["pipeline_apply_batch_experts"] > 0:
        summary["pipeline_apply_batch_activated_ratio"] = (
            summary["pipeline_apply_batch_activated"] / summary["pipeline_apply_batch_experts"]
        )
        summary["pipeline_apply_batch_warm_ratio"] = (
            summary["pipeline_apply_batch_warm"] / summary["pipeline_apply_batch_experts"]
        )
        summary["pipeline_apply_batch_cold_ratio"] = (
            summary["pipeline_apply_batch_cold"] / summary["pipeline_apply_batch_experts"]
        )
    else:
        summary["pipeline_apply_batch_activated_ratio"] = None
        summary["pipeline_apply_batch_warm_ratio"] = None
        summary["pipeline_apply_batch_cold_ratio"] = None
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
        activated_count = int(scheduler_summary.get("pipeline_promotion_source_activated", 0))
        warm_count = int(scheduler_summary.get("pipeline_promotion_source_warm", 0))
        cold_count = int(scheduler_summary.get("pipeline_promotion_source_cold", 0))
        promotion_total = activated_count + warm_count + cold_count
        runtime_apply_batch_count = int(
            scheduler_summary.get("offload_pipeline_apply_batch_count_total", 0)
        )
        runtime_apply_batch_experts = int(
            scheduler_summary.get("offload_pipeline_apply_batch_experts_total", 0)
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
            "pipeline_promotion_source_activated": activated_count,
            "pipeline_promotion_source_warm": warm_count,
            "pipeline_promotion_source_cold": cold_count,
            "pipeline_promotion_non_cold_total": activated_count + warm_count,
            "pipeline_promotion_non_cold_ratio": (
                (activated_count + warm_count) / promotion_total
                if promotion_total > 0
                else None
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
            "pipeline_apply_batch_activated": int(
                scheduler_summary.get("pipeline_apply_batch_activated", 0)
            ),
            "pipeline_apply_batch_warm": int(
                scheduler_summary.get("pipeline_apply_batch_warm", 0)
            ),
            "pipeline_apply_batch_cold": int(
                scheduler_summary.get("pipeline_apply_batch_cold", 0)
            ),
            "pipeline_apply_batch_activated_ratio": scheduler_summary.get(
                "pipeline_apply_batch_activated_ratio"
            ),
            "pipeline_apply_batch_warm_ratio": scheduler_summary.get(
                "pipeline_apply_batch_warm_ratio"
            ),
            "pipeline_apply_batch_cold_ratio": scheduler_summary.get(
                "pipeline_apply_batch_cold_ratio"
            ),
            "runtime_offload_pipeline_apply_batch_count_total": runtime_apply_batch_count,
            "runtime_offload_pipeline_apply_batch_experts_total": runtime_apply_batch_experts,
            "runtime_offload_pipeline_apply_batch_evictions_total": int(
                scheduler_summary.get("offload_pipeline_apply_batch_evictions_total", 0)
            ),
            "runtime_apply_batch_size_avg": (
                runtime_apply_batch_experts / runtime_apply_batch_count
                if runtime_apply_batch_count > 0
                else None
            ),
            "prepared_cache_limit": scheduler_summary.get("prepared_cache_limit"),
            "effective_prepared_cache_limit": scheduler_summary.get("effective_prepared_cache_limit"),
            "prepared_cache_size": scheduler_summary.get("prepared_cache_size"),
            "effective_warm_cache_limit": scheduler_summary.get("effective_warm_cache_limit"),
            "prepared_cache_utilization": scheduler_summary.get("prepared_cache_utilization"),
            "effective_prepared_cache_utilization": scheduler_summary.get(
                "effective_prepared_cache_utilization"
            ),
            "prepared_cache_rebalance_pressure_avg": scheduler_summary.get(
                "prepared_cache_rebalance_pressure_avg"
            ),
            "prepared_cache_rebalance_evicted_warm": int(
                scheduler_summary.get("prepared_cache_rebalance_evicted_warm", 0)
            ),
            "prepared_cache_rebalance_evicted_activated": int(
                scheduler_summary.get("prepared_cache_rebalance_evicted_activated", 0)
            ),
            "prepared_cache_rebalance_demoted_to_warm": int(
                scheduler_summary.get("prepared_cache_rebalance_demoted_to_warm", 0)
            ),
            "prepared_cache_rebalance_dropped_to_ready": int(
                scheduler_summary.get("prepared_cache_rebalance_dropped_to_ready", 0)
            ),
            "prepared_cache_rebalance_activated_ratio": scheduler_summary.get(
                "prepared_cache_rebalance_activated_ratio"
            ),
            "prepared_cache_activation_stage_bonus_avg": scheduler_summary.get(
                "prepared_cache_activation_stage_bonus_avg"
            ),
            "cold_promotion_penalty_avg": scheduler_summary.get("cold_promotion_penalty_avg"),
            "adaptive_activation_limit_avg": scheduler_summary.get(
                "adaptive_activation_limit_avg"
            ),
            "adaptive_prebuild_limit_avg": scheduler_summary.get(
                "adaptive_prebuild_limit_avg"
            ),
            "migration_activation_eviction_regressions": int(
                scheduler_summary.get("migration_activation_eviction_regressions", 0)
            ),
            "migration_warm_eviction_regressions": int(
                scheduler_summary.get("migration_warm_eviction_regressions", 0)
            ),
            "runtime_deferred_for_prefetch": int(
                scheduler_summary.get("runtime_deferred_for_prefetch", 0)
            ),
        }
        profiles.append(item)

        if decode_tps_avg is not None and decode_tps_avg > best_decode_tps:
            best_decode_tps = decode_tps_avg
            best_profile = item

    comparison_table = _build_profile_comparison_table(profiles)
    return {
        "sort_keys": list(PROFILE_SWEEP_SORT_KEYS),
        "metric_directions": dict(PROFILE_SWEEP_METRIC_DIRECTIONS),
        "profiles": profiles,
        "comparison_table": comparison_table,
        "best_by_metric": _summarize_best_profiles_by_metric(profiles),
        "best_by_decode_tokens_per_second": best_profile,
    }
