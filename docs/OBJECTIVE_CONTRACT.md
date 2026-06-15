# Objective Contract

## Stage 5.0 Objective Freeze

This contract freezes objective semantics only. It does not implement reward,
does not set scalar weights, does not train models, and does not migrate v5
code.

## Controlled Object

The controlled object is the application-level topology planning objective for
MARL-Topology after Stage 4.8 communication/consensus boundary audit.

The objective consumes registered evaluation quantities and diagnostics from
the Stage 3/4 stack. It is not an actor input schema, not a reward function, and
not a training loop.

## Inputs

Required evaluation inputs:

- `consensus_success_probability`: registered Stage 4 PBFT/application
  reliability probability.
- `protocol_latency`: protocol-level latency source quantity mapped to the
  registered metric concept `latency`.
- `protocol_energy`: protocol-level energy source quantity mapped to the
  registered metric concept `energy`.
- `topology_diagnostics`: registered diagnostic concept for edge count,
  per-primary reliability, retransmission attempts, feasibility tags, and
  similar non-objective fields.

The current implementation source for `protocol_latency` and `protocol_energy`
is Stage 4.6 protocol accounting. Future implementations must state the
accounting mode before comparison.

## Feasibility Condition

The reliability constraint is:

```text
consensus_success_probability >= tau_consensus
```

`tau_consensus` is a formal objective-contract parameter. It is not fixed by
Stage 5.0.

Stage 4.8 used `reliability_threshold = 0.2` only as an audit diagnostic
threshold to expose non-saturated boundary behavior. That value is not the
formal `tau_consensus` and must not be copied into reward, training, or final
evaluation without later scenario calibration or owner decision.

Boundary shorthand: not the formal `tau_consensus`.

Stage 5.0a adds the calibration plan in
`docs/TAU_CONSENSUS_CALIBRATION_PLAN.md`. That plan defines evidence and owner
decision requirements but still does not select the final `tau_consensus`.

Stage 5.0c adds the report design in
`docs/TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md`. That design specifies future
report tables and validation gates but still does not select the final
`tau_consensus`.

Stage 5.0d implements a report-only sensor in
`docs/STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md`. The implementation
requires owner-declared candidate tau values, uses Stage 4.8 rows as smoke-test
input only, and still does not select the final `tau_consensus`.

Stage 5.0e adds the fixture-family design in
`docs/STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md`. It specifies the
scenario coverage needed before calibration evidence can support owner threshold
selection.

Stage 5.0f implements the minimal alpha fixture suite in
`docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`. It creates executable
calibration evidence rows but still does not choose `tau_consensus`.

Stage 5.0g runs the report over the Stage 5.0f alpha fixture suite with
owner-supplied `tau = 0.9`, but still does not select final `tau_consensus`.

Stage 5.0h reframes `0.9` as a requirement baseline:

```text
tau_requirement_min = 0.9
```

This is a minimum acceptable consensus reliability requirement derived from
owner intent, not a fitted calibration value from the alpha fixture suite. If a
scenario is infeasible under `tau_requirement_min`, the next action is to
diagnose communication resources, topology candidates, protocol assumptions, or
simulation parameter realism before considering any lower diagnostic value.

Lower tau values may be used only as `tau_diagnostic_values` in reports, and
must not become final objective thresholds without explicit owner approval.

Stage 5.0m reviews objective readiness after Stage 5.0j, Stage 5.0k, and Stage
5.0l feasibility-envelope evidence. It allows the next task to be Stage 5.1 as
a reward implementation plan without code, while keeping final tau selection,
reward implementation, reward weights, training, model work, and v5 migration
blocked.

## Primary Optimization

Within the feasible region:

```text
minimize latency
minimize energy
```

The default interpretation is constrained multi-objective optimization:

1. reject infeasible topologies that fail `tau_consensus`;
2. compare feasible topologies by latency and energy;
3. keep Pareto trade-offs visible unless a future owner-approved contract
   chooses a scalarization or lexicographic rule.

Stage 5.0 does not choose latency/energy weights.

## Diagnostic And Tie-Breaker Fields

The following may be reported as diagnostics or deterministic tie-breakers only
after reliability feasibility and latency/energy semantics are unchanged:

- edge count;
- topology diagnostics;
- per-primary reliability;
- retransmission attempts;
- scheduled message counts;
- full-graph-baseline flags;
- oracle-candidate flags that are explicitly review-only.

Diagnostics must not become default reward components or primary objectives.
Oracle-candidate labels and evaluation-only diagnostics must not enter
deployment actor observations.

## Forbidden Objective Routes

The objective contract forbids:

- standalone timeout reward;
- standalone quorum reward;
- standalone density reward;
- standalone edge-count reward;
- reliability bonus above `tau_consensus`;
- full graph or full mask as an implicit objective;
- unregistered metrics;
- legacy v5 effective-success aliases;
- old reward formulas, weights, sentinel penalties, or phase scripts.

Timeout, quorum, and deadline already enter the system through Stage 3/4
communication and protocol contracts. They are not standalone objective terms.

## Metric Governance Impact

Stage 5.0 adds no new metric names.

The active registered metric concepts remain:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`protocol_latency` and `protocol_energy` are source quantities that map to
registered `latency` and `energy`; they are not new metric names unless a
future metric registry explicitly promotes them.

## Acceptance

- Reliability is a constraint, not an unlimited reward source.
- `tau_consensus` remains distinct from the Stage 4.8 audit threshold.
- `tau_consensus` selection must follow the Stage 5.0a calibration plan or an
  explicit owner decision.
- Latency and energy remain the primary optimization objectives inside the
  feasible region.
- Diagnostics remain diagnostics or explicitly declared tie-breakers.
- Reward implementation, reward weight calibration, and training remain
  blocked.
