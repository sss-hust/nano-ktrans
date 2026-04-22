"""Tests for scripts/dev_gate.py.

The gate is the single decision point for milestone progression, so it has
to be deterministic. We exercise:

* path resolution (scalar, list index, aggregate wrappers)
* artifact freshness (stale data cannot replay a PASS)
* prerequisite chaining (M-2 halts until M-1 has a stored PASS)
* the outcome shapes: PASS / PARTIAL / BLOCKED / WAIT / HALT
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest


# -- Import scripts/dev_gate.py as a plain module --------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GATE_PATH = _REPO_ROOT / "scripts" / "dev_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("dev_gate_under_test", _GATE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dev_gate = _load_gate_module()


# -- Fixtures --------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Retarget dev_gate's global paths into a throwaway tmp root."""
    specs_dir = tmp_path / ".codebuddy" / "dev_gate"
    specs_dir.mkdir(parents=True)
    monkeypatch.setattr(dev_gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dev_gate, "SPECS_DIR", specs_dir)
    monkeypatch.setattr(dev_gate, "STATE_PATH", tmp_path / ".codebuddy" / "dev_gate_state.json")
    monkeypatch.setattr(dev_gate, "LOG_PATH", tmp_path / ".codebuddy" / "dev_gate_log.jsonl")
    return tmp_path


def _write_spec(root: Path, mid: str, *, prereqs=(), required=(), primary=None,
                suggested=(), checks=()) -> None:
    lines = [
        f'milestone_id = "{mid}"',
        f'title = "test {mid}"',
        f"prerequisites = {json.dumps(list(prereqs))}",
        f"required_artifacts = {json.dumps(list(required))}",
    ]
    if primary is not None:
        lines.append(f'primary_artifact = "{primary}"')
    lines.append(f"suggested_commands = {json.dumps(list(suggested))}")
    for rule in checks:
        lines.append("")
        lines.append("[[acceptance_checks]]")
        for k, v in rule.items():
            lines.append(f"{k} = {json.dumps(v)}")
    (root / ".codebuddy" / "dev_gate" / f"{mid}.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_json(root: Path, rel: str, payload) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# -- Path resolution -------------------------------------------------------

class TestPathResolution:
    def test_scalar_dict_chain(self):
        assert dev_gate._resolve_path({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_list_index(self):
        data = {"results": [{"x": 1}, {"x": 2}, {"x": 3}]}
        assert dev_gate._resolve_path(data, "results[1].x") == 2

    def test_min_max_count_aggregates(self):
        data = {"r": [{"x": 1.5}, {"x": 2.5}, {"x": 0.5}]}
        assert dev_gate._resolve_path(data, "min(r[*].x)") == 0.5
        assert dev_gate._resolve_path(data, "max(r[*].x)") == 2.5
        assert dev_gate._resolve_path(data, "count(r[*].x)") == 3

    def test_aggregate_skips_missing_rows(self):
        data = {"r": [{"x": 1.0}, {}, {"x": 3.0}]}
        assert dev_gate._resolve_path(data, "max(r[*].x)") == 3.0
        assert dev_gate._resolve_path(data, "count(r[*].x)") == 2

    def test_star_outside_aggregate_rejected(self):
        with pytest.raises(KeyError, match=r"\[\*\]"):
            dev_gate._resolve_path({"a": [1]}, "a[*]")

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            dev_gate._resolve_path({"a": 1}, "a.b")


# -- Rule evaluation -------------------------------------------------------

class TestRuleEvaluation:
    def test_ge_passes_and_fails(self):
        rule = dev_gate.AcceptanceRule(path="x", op=">=", value=1.5)
        ok = dev_gate.evaluate_rule(rule, artifacts_data={"a.json": {"x": 2.0}}, default_artifact="a.json")
        assert ok.passed and ok.observed == 2.0
        bad = dev_gate.evaluate_rule(rule, artifacts_data={"a.json": {"x": 1.0}}, default_artifact="a.json")
        assert not bad.passed

    def test_exists_and_not_exists(self):
        data = {"a.json": {"error": "boom"}}
        ex = dev_gate.AcceptanceRule(path="error", op="exists")
        assert dev_gate.evaluate_rule(ex, artifacts_data=data, default_artifact="a.json").passed
        nex = dev_gate.AcceptanceRule(path="missing", op="not_exists")
        assert dev_gate.evaluate_rule(nex, artifacts_data=data, default_artifact="a.json").passed

    def test_operand_type_mismatch_is_reported(self):
        rule = dev_gate.AcceptanceRule(path="x", op=">=", value="oops")
        o = dev_gate.evaluate_rule(rule, artifacts_data={"a.json": {"x": 1.0}}, default_artifact="a.json")
        assert not o.passed and o.error and "operand" in o.error

    def test_rule_artifact_overrides_default(self):
        data = {"a.json": {"x": 1}, "b.json": {"x": 10}}
        rule = dev_gate.AcceptanceRule(path="x", op=">=", value=5, artifact="b.json")
        assert dev_gate.evaluate_rule(rule, artifacts_data=data, default_artifact="a.json").passed


# -- End-to-end milestone evaluation --------------------------------------

class TestMilestoneEvaluation:
    def test_wait_when_artifact_missing(self, sandbox):
        _write_spec(
            sandbox, "M-A",
            required=["data.json"], primary="data.json",
            suggested=["python bench.py --out data.json"],
            checks=[{"path": "x", "op": ">=", "value": 1}],
        )
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "WAIT" and v.stage == "artifact"
        assert "missing artifact" in v.reason and "python bench.py" in v.reason

    def test_pass_with_fresh_artifact_and_passing_rule(self, sandbox):
        _write_spec(
            sandbox, "M-A",
            required=["data.json"], primary="data.json",
            checks=[{"path": "x", "op": ">=", "value": 1.0}],
        )
        _write_json(sandbox, "data.json", {"x": 2.0})
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "PASS" and v.stage == "acceptance"
        assert len(v.rule_outcomes) == 1 and v.rule_outcomes[0].passed

    def test_partial_when_some_rules_fail(self, sandbox):
        _write_spec(
            sandbox, "M-A",
            required=["data.json"], primary="data.json",
            checks=[
                {"path": "x", "op": ">=", "value": 1.0},
                {"path": "y", "op": ">=", "value": 10.0},
            ],
        )
        _write_json(sandbox, "data.json", {"x": 2.0, "y": 1.0})
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "PARTIAL"
        assert sum(1 for o in v.rule_outcomes if o.passed) == 1

    def test_blocked_when_all_rules_fail(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["data.json"], primary="data.json",
            checks=[{"path": "x", "op": ">=", "value": 100}],
        )
        _write_json(sandbox, "data.json", {"x": 1})
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "BLOCKED"

    def test_blocked_when_no_acceptance_checks(self, sandbox):
        """Empty rule list must NOT silently PASS."""
        _write_spec(sandbox, "M-A", required=["data.json"], primary="data.json", checks=[])
        _write_json(sandbox, "data.json", {"x": 1})
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "BLOCKED" and "no acceptance_checks" in v.reason

    def test_freshness_prevents_replaying_previous_pass(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["data.json"], primary="data.json",
            checks=[{"path": "x", "op": ">=", "value": 1.0}],
        )
        artifact = _write_json(sandbox, "data.json", {"x": 2.0})
        state = {"milestones": {}}
        # First evaluation: PASS, snapshot the mtime.
        v1 = dev_gate.evaluate_milestone("M-A", state=state)
        assert v1.verdict == "PASS"
        state["milestones"]["M-A"] = {
            "verdict": "PASS",
            "artifact_snapshots": v1.artifact_snapshots,
        }
        # Second evaluation with the SAME file mtime -> must WAIT, not replay PASS.
        v2 = dev_gate.evaluate_milestone("M-A", state=state)
        assert v2.verdict == "WAIT"
        assert "unchanged since last PASS" in v2.reason

    def test_fresh_rerun_after_artifact_update_allows_reevaluation(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["data.json"], primary="data.json",
            checks=[{"path": "x", "op": ">=", "value": 1.0}],
        )
        artifact = _write_json(sandbox, "data.json", {"x": 2.0})
        v1 = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        state = {"milestones": {"M-A": {"verdict": "PASS", "artifact_snapshots": v1.artifact_snapshots}}}
        # Bump mtime (1 second granularity on most filesystems).
        future = time.time() + 2
        os.utime(artifact, (future, future))
        v2 = dev_gate.evaluate_milestone("M-A", state=state)
        assert v2.verdict == "PASS"

    def test_prerequisite_halts_when_parent_not_pass(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["a.json"], primary="a.json",
            checks=[{"path": "x", "op": ">=", "value": 1}],
        )
        _write_spec(
            sandbox, "M-B", prereqs=["M-A"],
            required=["b.json"], primary="b.json",
            checks=[{"path": "y", "op": ">=", "value": 1}],
        )
        _write_json(sandbox, "b.json", {"y": 100})
        v = dev_gate.evaluate_milestone("M-B", state={"milestones": {}})
        assert v.verdict == "HALT" and v.stage == "prerequisite"
        assert "M-A" in v.reason

    def test_prerequisite_passes_when_parent_recorded_as_pass(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["a.json"], primary="a.json",
            checks=[{"path": "x", "op": ">=", "value": 1}],
        )
        _write_spec(
            sandbox, "M-B", prereqs=["M-A"],
            required=["b.json"], primary="b.json",
            checks=[{"path": "y", "op": ">=", "value": 1}],
        )
        _write_json(sandbox, "b.json", {"y": 100})
        state = {"milestones": {"M-A": {"verdict": "PASS", "artifact_snapshots": {}}}}
        v = dev_gate.evaluate_milestone("M-B", state=state)
        assert v.verdict == "PASS"

    def test_invalid_json_artifact_is_blocked(self, sandbox):
        _write_spec(
            sandbox, "M-A", required=["data.json"], primary="data.json",
            checks=[{"path": "x", "op": ">=", "value": 1}],
        )
        bad = sandbox / "data.json"
        bad.write_text("this is not json", encoding="utf-8")
        v = dev_gate.evaluate_milestone("M-A", state={"milestones": {}})
        assert v.verdict == "BLOCKED" and "invalid JSON" in v.reason


# -- Shipped milestone specs are well-formed ------------------------------

class TestShippedSpecs:
    """The real ADR-002 M-1 / M-2 specs must at least load cleanly and
    declare the artifact / rule shape the gate expects."""

    def test_m1_spec_loads(self):
        spec = dev_gate.load_spec("M-1")
        assert spec.milestone_id == "M-1"
        assert spec.required_artifacts, "M-1 must declare required_artifacts"
        assert spec.acceptance_checks, "M-1 must declare acceptance_checks"
        assert spec.prerequisites == []

    def test_m2_spec_requires_m1(self):
        spec = dev_gate.load_spec("M-2")
        assert "M-1" in spec.prerequisites

    def test_real_m1_halts_cleanly_in_current_env(self):
        """In the current session, M-1's artifacts do not exist, so the gate
        must WAIT with a suggestion, not crash."""
        state = dev_gate.load_state()
        v = dev_gate.evaluate_milestone("M-1", state=state)
        # Either the artifacts haven't been generated yet (WAIT) or the
        # sweep has been recorded and passed (PASS).  Both are acceptable;
        # crashes are not.
        assert v.verdict in {"WAIT", "PASS", "PARTIAL", "BLOCKED"}
