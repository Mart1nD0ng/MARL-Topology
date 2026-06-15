# Stage 5.0f Minimal Tau Calibration Fixture Suite Implementation

## Scope

Stage 5.0f implements a minimal executable alpha fixture suite for
tau-consensus calibration reporting.

It does not select `tau_consensus`, does not implement reward, does not tune
weights, does not train models, does not add actor/critic/COMA/GNN/LSTM code,
and does not migrate v5 code.

Boundary shorthand: Stage 5.0f implements fixture rows, not tau selection.

## Reflection On Over-Conservatism

The previous recommendation inserted another plan-only step after the project
already had objective contracts, tau planning, report design, report
implementation, and fixture-family design. That was too conservative because
the controlled object had enough contract coverage and the missing actuator was
not more planning; it was an executable alpha sensor.

The corrected rule is: once contracts and gates cover the failure modes, prefer
the smallest executable implementation that creates new evidence. Add another
plan-only stage only when the next actuator is unsafe or underspecified.

## Controlled Object

The controlled object is the Stage 5 calibration evidence source consumed by
the Stage 5.0d tau-consensus report builder.

## Desired State

The project should have deterministic alpha rows covering representative
calibration regimes:

- clear free-space reference;
- near-threshold link budget;
- blocked or NLoS urban communication;
- same-resource interference;
- tight deadline and retransmission pressure;
- unreachable reliability target;
- sparse-vs-dense topology trade-off;
- weak-primary PBFT distribution.

Each family provides at least:

- weak or disconnected baseline;
- sparse candidate;
- dense full-graph baseline.

## Implementation

The executable suite is implemented in:

```text
src/marl_topology/evaluation/calibration_fixture_suite.py
```

The replay script is:

```powershell
python scripts/replay/tau_consensus_fixture_suite_report.py
```

The Stage 5.0d report CLI can consume the alpha suite explicitly:

```powershell
python scripts/replay/tau_consensus_calibration_report.py --source stage5_0f --tau 0.05 --tau 0.5
```

The tau values in this command are caller-supplied examples, not defaults.

The suite emits rows with:

- `scenario_set_id`
- `scenario_family`
- `scenario_id`
- `fixture_id`
- `topology_name`
- `topology_family`
- `selected_edge_count`
- `is_full_graph_baseline`
- `is_oracle_candidate`
- `is_deployment_actor_input`
- `consensus_success_probability`
- `latency`
- `energy`
- `per_primary_reliability`
- `diagnostic_flags`

Additional diagnostic fields such as `coverage_axis` are allowed but are not
new core metrics.

## Stage 5.0d Report Integration

`build_tau_consensus_calibration_report(...)` now accepts:

- Stage 4.8 smoke-test source reports;
- Stage 5.0f alpha fixture suite reports.

When the source is Stage 5.0f, the report uses the suite's
`scenario_manifest`, `topology_evaluation_rows`, and `protocol_regime_rows`.
It still leaves `final_tau_consensus`, `recommended_tau_candidate`, and
`owner_selected_tau` unset.

## Metric Governance

Stage 5.0f adds no metric names.

Metric-valued row fields map to:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Family ids, topology variants, diagnostic flags, coverage axes, and feasibility
views are diagnostics or identifiers.

## Exit Criteria Coverage

- Non-saturated consensus rows exist.
- Sparse candidates and dense full-graph baselines exist in every family.
- At least one sparse candidate has lower latency and energy than the
  full-graph baseline.
- At least one full graph is worse than the sparse candidate due to
  interference or resource cost.
- At least one failed scheduled or unreachable target row has positive latency
  or energy.
- Weak-primary per-primary reliability spread exists.
- Stage 5.0d report builder consumes the new fixture rows.

## Boundaries

- No final tau is selected.
- No candidate tau is provided as a default.
- Full graph is a baseline, not an oracle.
- Oracle labels remain review-only diagnostics; the alpha suite currently does
  not emit oracle candidates.
- No actor/deployment input receives oracle or calibration labels.
- No reward implementation or training code is added.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

- The alpha suite is deterministic and representative, not yet a full-scale
  calibrated city distribution.
- Some rows use declared alpha message probabilities to stress report behavior;
  future hardening can attach each family to richer Stage 3 geometry/channel
  builders.
- Owner-supplied candidate tau values are still required before a calibration
  report run.
