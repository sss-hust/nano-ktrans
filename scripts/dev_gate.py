#!/usr/bin/env python3
"""
dev_gate — data-driven gate for milestone progression.

Problem
-------
ADR-002 splits the work into four milestones (M-1 .. M-4).  Each milestone
produces one or more benchmark JSON files that must meet concrete numeric
acceptance criteria before the next milestone is allowed to start.  Without
an explicit gate, the project drifts: stale numbers from a previous sweep
get treated as if they were the current milestone's evidence, or new work
gets started before the previous milestone's KPI was actually met.

This script is the single place where milestone progression is decided.
It refuses to say "go" unless it has evidence — *fresh* evidence — that
the previous milestone passed.

Pipeline
--------
stage 1  prerequisite_check
         For every milestone listed in `prerequisites`, re-check that
         milestone's own acceptance rules against its artifacts.  If any
         prerequisite fails, HALT here and report which one.

stage 2  artifact_check
         All `required_artifacts` must exist on disk AND be newer than
         the last gate evaluation for this milestone.  If a file is
         missing, report the suggested command to generate it.  If a
         file has not been updated since last evaluation, WAIT — the
         previous sweep is what we already judged and cannot be reused.

stage 3  acceptance_check
         Run each `acceptance_checks` rule against the fresh artifacts.
         A rule is a dict: { path: "dot.path.into.json", op: ">=", value: 1.5 }.
         All rules must pass.  Mixed outcomes are reported as PARTIAL.

Determinism & audit trail
-------------------------
Every evaluation appends a line to `.codebuddy/dev_gate_log.jsonl` with
{timestamp, milestone_id, verdict, details}, and upserts the summary
state into `.codebuddy/dev_gate_state.json`.  `--force` is allowed but
records bypassed=true so PR review can spot it.

This script intentionally has **no runtime dependency on nano_ktrans**.
It reads only the JSON files the benchmark scripts produce.  That way it
stays usable even when the rest of the codebase is mid-refactor.

CLI
---
    python scripts/dev_gate.py check           # evaluate all milestones
    python scripts/dev_gate.py check M-2       # evaluate one milestone
    python scripts/dev_gate.py status          # print cached state only
    python scripts/dev_gate.py bless M-1 --force --note "manual override: ..."

Spec format
-----------
Milestone specs live in `.codebuddy/dev_gate/<milestone_id>.toml`.  See
that directory's README.md for fields and examples.  Shipping today:

    .codebuddy/dev_gate/M-1.toml
    .codebuddy/dev_gate/M-2.toml   # inert until M-2 starts; included as template
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / ".codebuddy" / "dev_gate"
STATE_PATH = REPO_ROOT / ".codebuddy" / "dev_gate_state.json"
LOG_PATH = REPO_ROOT / ".codebuddy" / "dev_gate_log.jsonl"


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceRule:
    """A single numeric / equality rule against an artifact JSON.

    `path` uses dot notation into the artifact.  Indexing is allowed with
    `[N]` syntax, e.g. `results[0].pim_vs_cpu_grouped_ratio`.  A special
    aggregate `path` of the form `min(results[*].x)` / `max(...)` /
    `count(...)` is supported for coarse bucket checks.
    """
    path: str
    op: str                    # one of: ==, !=, <, <=, >, >=, exists, not_exists
    value: Any = None
    artifact: str | None = None  # defaults to spec.primary_artifact if unset
    reason: str = ""           # human-readable rationale, shown in reports


@dataclass
class MilestoneSpec:
    milestone_id: str
    title: str
    prerequisites: list[str] = field(default_factory=list)
    required_artifacts: list[str] = field(default_factory=list)
    primary_artifact: str | None = None
    suggested_commands: list[str] = field(default_factory=list)
    acceptance_checks: list[AcceptanceRule] = field(default_factory=list)
    next_tasks: list[str] = field(default_factory=list)


@dataclass
class RuleOutcome:
    rule: AcceptanceRule
    passed: bool
    observed: Any = None
    error: str | None = None


@dataclass
class MilestoneVerdict:
    milestone_id: str
    stage: str                 # prerequisite / artifact / acceptance
    verdict: str               # PASS / PARTIAL / BLOCKED / WAIT / HALT
    reason: str
    artifact_snapshots: dict[str, float] = field(default_factory=dict)  # path -> mtime
    rule_outcomes: list[RuleOutcome] = field(default_factory=list)
    bypassed: bool = False
    timestamp: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
#  Spec loading
# ---------------------------------------------------------------------------

def load_spec(milestone_id: str) -> MilestoneSpec:
    path = SPECS_DIR / f"{milestone_id}.toml"
    if not path.exists():
        raise FileNotFoundError(f"No gate spec for {milestone_id} at {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    checks_raw = raw.get("acceptance_checks", [])
    checks = [
        AcceptanceRule(
            path=item["path"],
            op=item["op"],
            value=item.get("value"),
            artifact=item.get("artifact"),
            reason=item.get("reason", ""),
        )
        for item in checks_raw
    ]
    return MilestoneSpec(
        milestone_id=raw["milestone_id"],
        title=raw["title"],
        prerequisites=list(raw.get("prerequisites", [])),
        required_artifacts=list(raw.get("required_artifacts", [])),
        primary_artifact=raw.get("primary_artifact"),
        suggested_commands=list(raw.get("suggested_commands", [])),
        acceptance_checks=checks,
        next_tasks=list(raw.get("next_tasks", [])),
    )


def list_spec_ids() -> list[str]:
    if not SPECS_DIR.exists():
        return []
    return sorted(p.stem for p in SPECS_DIR.glob("*.toml"))


# ---------------------------------------------------------------------------
#  State persistence
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"milestones": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"milestones": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def append_log(verdict: MilestoneVerdict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = asdict(verdict)
    # asdict cannot see nested dataclasses inside lists consistently across
    # py versions, rebuild the rule_outcomes list explicitly.
    entry["rule_outcomes"] = [
        {
            "rule": asdict(o.rule),
            "passed": o.passed,
            "observed": o.observed,
            "error": o.error,
        }
        for o in verdict.rule_outcomes
    ]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
#  Path & rule evaluation
# ---------------------------------------------------------------------------

def _resolve_path(data: Any, path: str) -> Any:
    """Minimal JSON path resolver supporting `a.b[2].c` and
    aggregate wrappers `min(a.b[*].c)`, `max(...)`, `count(...)`.
    """
    path = path.strip()

    # Aggregate forms: fn(inner)
    for fn_name, fn in (("min", min), ("max", max), ("count", len)):
        prefix = fn_name + "("
        if path.startswith(prefix) and path.endswith(")"):
            inner = path[len(prefix) : -1]
            values = _resolve_glob(data, inner)
            if fn_name == "count":
                return len(values)
            filtered = [v for v in values if isinstance(v, (int, float))]
            if not filtered:
                raise KeyError(f"Aggregate {fn_name}: no numeric values at {inner!r}")
            return fn(filtered)

    # Non-aggregate: single scalar lookup.
    return _resolve_scalar(data, path)


def _split_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
        elif ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.index("]", i)
            tokens.append(path[i : end + 1])  # keep brackets for identification
            i = end
        else:
            buf += ch
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


def _resolve_scalar(data: Any, path: str) -> Any:
    if not path:
        return data
    cur = data
    for token in _split_tokens(path):
        if token.startswith("[") and token.endswith("]"):
            idx_raw = token[1:-1]
            if idx_raw == "*":
                raise KeyError(
                    f"'[*]' globs only allowed inside min()/max()/count(); got {path!r}"
                )
            idx = int(idx_raw)
            if not isinstance(cur, list):
                raise KeyError(f"Expected list at token {token!r} in {path!r}")
            cur = cur[idx]
        else:
            if not isinstance(cur, dict):
                raise KeyError(f"Expected dict at token {token!r} in {path!r}")
            if token not in cur:
                raise KeyError(f"Missing key {token!r} in {path!r}")
            cur = cur[token]
    return cur


def _resolve_glob(data: Any, path: str) -> list[Any]:
    """Resolve a single `[*]` glob and return the resulting list."""
    tokens = _split_tokens(path)
    if sum(1 for t in tokens if t == "[*]") != 1:
        raise KeyError(f"Aggregate path must have exactly one [*] wildcard: {path!r}")
    before: list[str] = []
    after: list[str] = []
    seen_star = False
    for t in tokens:
        if t == "[*]":
            seen_star = True
            continue
        (after if seen_star else before).append(t)
    before_path = ".".join(t if not t.startswith("[") else t for t in before).replace(".[", "[")
    after_path = ".".join(t if not t.startswith("[") else t for t in after).replace(".[", "[")
    collection = _resolve_scalar(data, before_path)
    if not isinstance(collection, list):
        raise KeyError(f"[*] points at a non-list for path {path!r}")
    values: list[Any] = []
    for item in collection:
        try:
            values.append(_resolve_scalar(item, after_path))
        except KeyError:
            # Skip rows that lack the field; acceptance rules use these for
            # "most rows must exceed X" semantics, not "all rows exist".
            continue
    return values


_OPS: dict[str, Any] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def evaluate_rule(
    rule: AcceptanceRule,
    *,
    artifacts_data: dict[str, Any],
    default_artifact: str | None,
) -> RuleOutcome:
    artifact_key = rule.artifact or default_artifact
    if artifact_key is None:
        return RuleOutcome(rule, False, error="rule.artifact unset and spec has no primary_artifact")
    data = artifacts_data.get(artifact_key)
    if data is None:
        return RuleOutcome(rule, False, error=f"artifact {artifact_key!r} not loaded")

    # Existence operators don't traverse to a value.
    if rule.op in ("exists", "not_exists"):
        try:
            _ = _resolve_path(data, rule.path)
            return RuleOutcome(rule, rule.op == "exists", observed="<present>")
        except KeyError:
            return RuleOutcome(rule, rule.op == "not_exists", observed="<absent>")

    try:
        observed = _resolve_path(data, rule.path)
    except KeyError as exc:
        return RuleOutcome(rule, False, error=str(exc))

    op_fn = _OPS.get(rule.op)
    if op_fn is None:
        return RuleOutcome(rule, False, observed=observed, error=f"unknown op {rule.op!r}")
    try:
        passed = bool(op_fn(observed, rule.value))
    except TypeError as exc:
        return RuleOutcome(rule, False, observed=observed, error=f"operand type mismatch: {exc}")
    return RuleOutcome(rule, passed, observed=observed)


# ---------------------------------------------------------------------------
#  The three gate stages
# ---------------------------------------------------------------------------

def _verdict(
    milestone_id: str,
    stage: str,
    result: str,
    reason: str,
    snapshots: dict[str, float] | None = None,
    outcomes: list[RuleOutcome] | None = None,
    notes: str = "",
) -> MilestoneVerdict:
    return MilestoneVerdict(
        milestone_id=milestone_id,
        stage=stage,
        verdict=result,
        reason=reason,
        artifact_snapshots=snapshots or {},
        rule_outcomes=outcomes or [],
        timestamp=_now_iso(),
        notes=notes,
    )


def _stage_artifact_check(
    spec: MilestoneSpec,
    state: dict[str, Any],
) -> tuple[MilestoneVerdict | None, dict[str, float], dict[str, Any]]:
    """Returns (halt_verdict_or_None, snapshot_mtimes, loaded_artifacts)."""
    snapshots: dict[str, float] = {}
    loaded: dict[str, Any] = {}
    missing: list[str] = []
    stale: list[str] = []

    prev_snapshots: dict[str, float] = (
        (state.get("milestones", {}).get(spec.milestone_id) or {}).get("artifact_snapshots") or {}
    )

    for artifact in spec.required_artifacts:
        abs_path = REPO_ROOT / artifact
        if not abs_path.exists():
            missing.append(artifact)
            continue
        mtime = abs_path.stat().st_mtime
        snapshots[artifact] = mtime
        # Freshness: strictly newer than the snapshot we evaluated last PASS,
        # otherwise we'd be re-judging the same data forever.
        prev_mtime = prev_snapshots.get(artifact)
        if prev_mtime is not None and mtime <= prev_mtime:
            stale.append(artifact)
        try:
            loaded[artifact] = json.loads(abs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return (
                _verdict(
                    spec.milestone_id, "artifact", "BLOCKED",
                    reason=f"{artifact}: invalid JSON ({exc})",
                    snapshots=snapshots,
                ),
                snapshots,
                loaded,
            )

    if missing:
        cmds = "\n  ".join(spec.suggested_commands) if spec.suggested_commands else "(no commands registered)"
        return (
            _verdict(
                spec.milestone_id, "artifact", "WAIT",
                reason=(
                    f"missing artifact(s): {missing}.\n"
                    f"Suggested commands to generate them:\n  {cmds}"
                ),
                snapshots=snapshots,
            ),
            snapshots,
            loaded,
        )
    if stale:
        return (
            _verdict(
                spec.milestone_id, "artifact", "WAIT",
                reason=(
                    f"artifact(s) unchanged since last PASS: {stale}.\n"
                    f"Re-run the benchmark so the previous verdict is not replayed."
                ),
                snapshots=snapshots,
            ),
            snapshots,
            loaded,
        )
    return None, snapshots, loaded


def _stage_acceptance_check(
    spec: MilestoneSpec,
    *,
    loaded: dict[str, Any],
    snapshots: dict[str, float],
) -> MilestoneVerdict:
    outcomes = [
        evaluate_rule(rule, artifacts_data=loaded, default_artifact=spec.primary_artifact)
        for rule in spec.acceptance_checks
    ]
    if not outcomes:
        return _verdict(
            spec.milestone_id, "acceptance", "BLOCKED",
            reason="spec has no acceptance_checks; refusing to PASS silently",
            snapshots=snapshots,
            outcomes=outcomes,
        )
    passed_n = sum(1 for o in outcomes if o.passed)
    total_n = len(outcomes)
    if passed_n == total_n:
        return _verdict(
            spec.milestone_id, "acceptance", "PASS",
            reason=f"all {total_n} acceptance rules satisfied",
            snapshots=snapshots, outcomes=outcomes,
        )
    if passed_n == 0:
        return _verdict(
            spec.milestone_id, "acceptance", "BLOCKED",
            reason=f"0/{total_n} acceptance rules passed",
            snapshots=snapshots, outcomes=outcomes,
        )
    return _verdict(
        spec.milestone_id, "acceptance", "PARTIAL",
        reason=f"only {passed_n}/{total_n} acceptance rules passed",
        snapshots=snapshots, outcomes=outcomes,
    )


def evaluate_milestone(milestone_id: str, state: dict[str, Any]) -> MilestoneVerdict:
    spec = load_spec(milestone_id)

    # stage 1: prerequisites.
    for prereq_id in spec.prerequisites:
        prereq_state = (state.get("milestones", {}) or {}).get(prereq_id)
        if not prereq_state or prereq_state.get("verdict") != "PASS":
            return _verdict(
                spec.milestone_id, "prerequisite", "HALT",
                reason=(
                    f"prerequisite {prereq_id} has not passed the gate yet "
                    f"(run `dev_gate check {prereq_id}` first)"
                ),
            )

    # stage 2: artifact presence + freshness.
    halt, snapshots, loaded = _stage_artifact_check(spec, state)
    if halt is not None:
        return halt

    # stage 3: numeric acceptance.
    return _stage_acceptance_check(spec, loaded=loaded, snapshots=snapshots)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _render_verdict(v: MilestoneVerdict) -> str:
    lines = [
        f"[{v.verdict}] {v.milestone_id}  (stage={v.stage})",
        f"    reason: {v.reason}",
    ]
    if v.bypassed:
        lines.append("    bypassed: TRUE (--force was used)")
    if v.artifact_snapshots:
        lines.append("    artifacts:")
        for p, mt in sorted(v.artifact_snapshots.items()):
            lines.append(f"      - {p}  (mtime={datetime.fromtimestamp(mt, tz=timezone.utc).isoformat(timespec='seconds')})")
    if v.rule_outcomes:
        lines.append("    rules:")
        for o in v.rule_outcomes:
            mark = "✓" if o.passed else "✗"
            desc = f"{o.rule.path} {o.rule.op} {o.rule.value!r}"
            if o.error:
                suffix = f"  ERROR: {o.error}"
            else:
                suffix = f"  observed={o.observed!r}"
            reason = f"  [{o.rule.reason}]" if o.rule.reason else ""
            lines.append(f"      {mark} {desc}{suffix}{reason}")
    return "\n".join(lines)


def _persist(state: dict[str, Any], verdict: MilestoneVerdict) -> None:
    state.setdefault("milestones", {})
    state["milestones"][verdict.milestone_id] = {
        "verdict": verdict.verdict,
        "stage": verdict.stage,
        "reason": verdict.reason,
        "timestamp": verdict.timestamp,
        "bypassed": verdict.bypassed,
        "notes": verdict.notes,
        # Only bless artifact mtimes as 'the snapshot we trust' when we PASSED.
        # WAIT / PARTIAL / BLOCKED must re-trigger on the next sweep.
        "artifact_snapshots": verdict.artifact_snapshots if verdict.verdict == "PASS" else {},
    }
    state["last_updated"] = _now_iso()
    save_state(state)
    append_log(verdict)


def cmd_check(args: argparse.Namespace) -> int:
    state = load_state()
    milestone_ids: Iterable[str] = args.milestones or list_spec_ids()
    exit_code = 0
    for mid in milestone_ids:
        verdict = evaluate_milestone(mid, state)
        _persist(state, verdict)
        print(_render_verdict(verdict))
        if verdict.verdict != "PASS":
            exit_code = 1
            if args.fail_fast:
                break
        print()  # blank line between milestones
    return exit_code


def cmd_status(_: argparse.Namespace) -> int:
    state = load_state()
    milestones = state.get("milestones", {})
    if not milestones:
        print("dev_gate: no milestones evaluated yet.")
        return 0
    print(f"dev_gate state (last_updated={state.get('last_updated')}):")
    for mid in sorted(milestones.keys()):
        info = milestones[mid]
        flag = "(bypassed)" if info.get("bypassed") else ""
        print(f"  {mid}: {info.get('verdict')}  at={info.get('timestamp')}  {flag}")
        reason = info.get("reason", "")
        if reason:
            print(f"    reason: {reason}")
    return 0


def cmd_bless(args: argparse.Namespace) -> int:
    if not args.force:
        print("refusing to bless without --force; blessing is auditable.", file=sys.stderr)
        return 2
    if not args.note:
        print("--note is required with --force (leave a rationale for the audit log).", file=sys.stderr)
        return 2
    state = load_state()
    spec = load_spec(args.milestone)
    snapshots: dict[str, float] = {}
    for artifact in spec.required_artifacts:
        p = REPO_ROOT / artifact
        if p.exists():
            snapshots[artifact] = p.stat().st_mtime
    verdict = _verdict(
        args.milestone, "manual", "PASS",
        reason=f"manually blessed: {args.note}",
        snapshots=snapshots,
        notes=args.note,
    )
    verdict.bypassed = True
    _persist(state, verdict)
    print(_render_verdict(verdict))
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    ids = list_spec_ids()
    if not ids:
        print(f"no specs under {SPECS_DIR}")
        return 1
    for mid in ids:
        spec = load_spec(mid)
        prereqs = ", ".join(spec.prerequisites) or "(none)"
        print(f"{mid}  —  {spec.title}")
        print(f"    prerequisites: {prereqs}")
        print(f"    required_artifacts:")
        for a in spec.required_artifacts:
            print(f"      - {a}")
        print(f"    acceptance_checks: {len(spec.acceptance_checks)} rule(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="data-driven milestone gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="evaluate one or more milestone gates")
    p_check.add_argument("milestones", nargs="*", help="milestone IDs, empty = all")
    p_check.add_argument("--fail-fast", action="store_true",
                         help="stop at the first non-PASS milestone")
    p_check.set_defaults(func=cmd_check)

    p_status = sub.add_parser("status", help="print cached gate state")
    p_status.set_defaults(func=cmd_status)

    p_bless = sub.add_parser("bless", help="mark a milestone as PASS manually (auditable)")
    p_bless.add_argument("milestone")
    p_bless.add_argument("--force", action="store_true",
                         help="required; refuses to proceed otherwise")
    p_bless.add_argument("--note", default="",
                         help="required rationale; stored in the audit log")
    p_bless.set_defaults(func=cmd_bless)

    p_list = sub.add_parser("list", help="list registered milestone specs")
    p_list.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
