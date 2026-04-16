from __future__ import annotations

from dataclasses import replace
from typing import Any

from .dynamic_expert_scheduler import SchedulerConfig


SCHEDULER_PROFILE_BASELINE = "baseline"
SCHEDULER_PROFILE_OVERLAP_SAFE = "overlap_safe"
SCHEDULER_PROFILE_EAGER = "eager"

SCHEDULER_PROFILE_NAMES = (
    SCHEDULER_PROFILE_BASELINE,
    SCHEDULER_PROFILE_OVERLAP_SAFE,
    SCHEDULER_PROFILE_EAGER,
)


def normalize_scheduler_profiles(
    profiles: list[str] | tuple[str, ...] | None,
    *,
    default_profile: str = SCHEDULER_PROFILE_BASELINE,
) -> list[str]:
    requested = list(profiles) if profiles else [default_profile]
    normalized: list[str] = []
    seen: set[str] = set()
    for profile in requested:
        candidate = (profile or default_profile).strip().lower().replace("-", "_")
        if candidate not in SCHEDULER_PROFILE_NAMES:
            raise ValueError(
                f"Unsupported scheduler profile: {profile}. "
                f"Available profiles: {', '.join(SCHEDULER_PROFILE_NAMES)}"
            )
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def resolve_scheduler_profile(
    profile: str | None,
    *,
    base_config: SchedulerConfig,
) -> SchedulerConfig:
    normalized = (profile or SCHEDULER_PROFILE_BASELINE).strip().lower().replace("-", "_")
    normalized = normalize_scheduler_profiles([normalized])[0]

    if normalized == SCHEDULER_PROFILE_BASELINE:
        return replace(
            base_config,
            prefill_collect_only=True,
            step_stride_prefill=max(8, int(base_config.step_stride_prefill)),
            step_stride_decode=1,
            decode_require_prefetch_ready=False,
            prefetch_candidate_budget_per_layer=max(
                int(base_config.prefetch_candidate_budget_per_layer),
                0,
            ),
        )

    if normalized == SCHEDULER_PROFILE_OVERLAP_SAFE:
        return replace(
            base_config,
            prefill_collect_only=True,
            step_stride_prefill=max(8, int(base_config.step_stride_prefill)),
            step_stride_decode=1,
            decode_require_prefetch_ready=True,
            demotion_idle_steps=max(2, int(base_config.demotion_idle_steps)),
            migration_cooldown_steps=max(2, int(base_config.migration_cooldown_steps)),
            prefetch_candidate_budget_per_layer=max(
                int(base_config.prefetch_candidate_budget_per_layer),
                2,
            ),
        )

    return replace(
        base_config,
        prefill_collect_only=False,
        step_stride_prefill=min(4, max(1, int(base_config.step_stride_prefill))),
        step_stride_decode=1,
        decode_require_prefetch_ready=False,
        demotion_idle_steps=max(1, int(base_config.demotion_idle_steps)),
        migration_cooldown_steps=max(1, int(base_config.migration_cooldown_steps)),
        prefetch_candidate_budget_per_layer=max(
            int(base_config.prefetch_candidate_budget_per_layer),
            4,
        ),
    )


def apply_scheduler_overrides(
    config: SchedulerConfig,
    **overrides: Any,
) -> SchedulerConfig:
    filtered = {key: value for key, value in overrides.items() if value is not None}
    if not filtered:
        return config
    return replace(config, **filtered)


def resolve_prepared_cache_budget(profile: str, config: SchedulerConfig) -> int:
    normalized = normalize_scheduler_profiles([profile])[0]
    base_budget = max(
        int(config.decode_promote_k) * 2,
        int(config.prefetch_candidate_budget_per_layer),
        2,
    )

    if normalized == SCHEDULER_PROFILE_OVERLAP_SAFE:
        return max(base_budget + 1, int(config.decode_promote_k) * 2 + 1)
    if normalized == SCHEDULER_PROFILE_EAGER:
        return max(
            base_budget + 2,
            int(config.decode_promote_k) * 2 + 2,
            int(config.prefetch_candidate_budget_per_layer) + 1,
        )
    return base_budget


def scheduler_profile_summary(profile: str, config: SchedulerConfig) -> dict[str, Any]:
    prepared_cache_budget = resolve_prepared_cache_budget(profile, config)
    return {
        "profile": profile,
        "prefill_collect_only": bool(config.prefill_collect_only),
        "step_stride_prefill": int(config.step_stride_prefill),
        "step_stride_decode": int(config.step_stride_decode),
        "demotion_idle_steps": int(config.demotion_idle_steps),
        "migration_cooldown_steps": int(config.migration_cooldown_steps),
        "decode_require_prefetch_ready": bool(config.decode_require_prefetch_ready),
        "prefetch_candidate_budget_per_layer": int(config.prefetch_candidate_budget_per_layer),
        "prepared_cache_budget_heuristic": int(prepared_cache_budget),
    }
