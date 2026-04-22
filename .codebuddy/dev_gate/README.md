# dev_gate specs

Each `<milestone_id>.toml` is read by `scripts/dev_gate.py`.  The gate
refuses to let the project move on to the next milestone unless:

1. every milestone in `prerequisites` has a `PASS` verdict in
   `.codebuddy/dev_gate_state.json`,
2. every path in `required_artifacts` exists and is **strictly newer**
   than the mtime snapshot that was blessed on the last PASS
   (prevents stale data from being replayed as fresh evidence),
3. every rule in `acceptance_checks` is satisfied against the fresh
   artifacts.

## Spec fields

```toml
milestone_id       = "M-1"                  # required, matches filename
title              = "..."                  # one-line description
prerequisites      = ["M-0"]                # optional, list of milestone_ids
required_artifacts = [                      # required, project-relative paths
  "benchmarks/results/...json",
]
primary_artifact   = "benchmarks/results/...json"   # optional; default for rule.artifact
suggested_commands = [                      # optional, printed on WAIT
  "python benchmarks/foo.py --json-out ...",
]
next_tasks         = ["..."]                # optional, printed on PASS

[[acceptance_checks]]
path     = "results[0].pim_vs_cpu_grouped_ratio"
op       = ">="
value    = 1.5
reason   = "ADR-002 KPI: PIM >= 1.5x CPU grouped"
artifact = "benchmarks/results/...json"    # optional; overrides primary_artifact
```

## Path syntax

* `a.b.c`            – dict traversal
* `a.items[2].x`     – positional indexing
* `min(a.items[*].x)`, `max(...)`, `count(...)`
  – aggregate over a list.  Exactly one `[*]` wildcard is allowed.

## Ops

`==`, `!=`, `<`, `<=`, `>`, `>=`, `exists`, `not_exists`.

## Notes

* A WAIT verdict is *not* a failure.  It just means the sweep has not
  been re-run since the last PASS.  The gate refuses to re-evaluate
  stale data so the verdict cannot be laundered.
* `dev_gate bless <id> --force --note "..."` is the only way to bypass
  a check.  The override is recorded in `dev_gate_state.json` with
  `bypassed = true` and in `dev_gate_log.jsonl` as a new entry.
