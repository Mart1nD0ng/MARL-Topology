# Stage 5.0d Tau Consensus Calibration Report

## Scope

Stage 5.0d implements a report-only tau-consensus calibration sensor without
tau selection.

Boundary shorthand: Stage 5.0d is implemented without tau selection.

It consumes Stage 4.8 smoke-test rows and owner-declared candidate tau values.
It does not select `tau_consensus`, does not implement reward, does not tune
weights, does not train models, does not add actor/critic code, and does not
migrate v5 code.

Stage 4.8 smoke-test rows are boundary sensors, not a final calibration set.

## Controlled Object

The controlled object is the executable report interface that converts
registered communication/consensus evidence into a tau feasibility view.

## Desired State

The report should make threshold discussion auditable while keeping ownership of
the final threshold outside Codex automation:

- candidate tau values are explicitly supplied by the caller;
- no default numeric tau candidate is provided;
- Stage 4.8 `reliability_threshold = 0.2` is not copied as a default;
- full graph remains a baseline, not an oracle;
- oracle labels are not actor inputs;
- feasibility is separated from latency and energy optimization;
- owner decision remains required before final tau selection.

## Inputs

The implementation currently accepts:

- owner-declared candidate tau values;
- optional candidate source label and owner note;
- Stage 4.8 boundary audit report rows, either supplied directly or built from
  the deterministic Stage 4.8 sensor.

The command-line interface requires at least one tau value:

```powershell
python scripts/replay/tau_consensus_calibration_report.py --tau 0.15 --tau 0.55
```

The example values are command inputs only. They are not project defaults.

## Outputs

The report prints a JSON payload with:

- `scenario_manifest`;
- `topology_evaluation_rows`;
- `candidate_tau_rows`;
- `protocol_regime_rows`;
- `scenario_summary`;
- `tau_feasibility_summary`;
- `topology_by_tau_detail`;
- `owner_decision_packet`;
- `metric_governance`;
- `checks`.

`owner_decision_packet.recommended_tau_candidate`,
`owner_decision_packet.owner_selected_tau`, and `final_tau_consensus` remain
unset in Stage 5.0d.

## Metric Governance

Stage 5.0d adds no metric names.

Metric-valued fields map to existing registered concepts:

- `consensus_success_probability`;
- `latency`;
- `energy`;
- `topology_diagnostics`.

Feasibility flags, Pareto counts, tau candidate rows, and owner decision fields
are report diagnostics or identifiers, not new core metrics.

## Validation Gates

The report must fail if:

- no candidate tau values are supplied;
- any candidate tau is outside `[0, 1]`;
- a candidate is marked as a default;
- a candidate is marked as selected;
- the source is not the expected Stage 4.8 report shape;
- full graph is treated as oracle;
- oracle labels are deployment actor inputs;
- unregistered metric-valued fields are introduced.

## Deferred

- final `tau_consensus` selection;
- scenario-family calibration set design;
- large-scale calibration runs;
- reward implementation;
- reward weight calibration;
- training;
- actor/critic/COMA/GNN/LSTM work.

## Residual Risks

- Stage 4.8 remains a small deterministic boundary sensor, not a final scenario
  calibration suite.
- The report summarizes feasibility for supplied tau candidates but does not
  recommend one.
- Future scenario-family coverage must be designed before treating the report as
  calibration evidence for final threshold selection.

## Stage 5.0e / 5.0f Follow-Up

Stage 5.0e is documented in:

`docs/STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md`

It defines the fixture-family coverage that a future calibration manifest should
provide before Stage 5.0d-style reporting is treated as calibration evidence.
It still does not run calibration or select `tau_consensus`.

Stage 5.0f is documented in:

`docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`

Stage 5.0f extends this report builder to consume the executable alpha fixture
suite as a source report. It still does not select `tau_consensus`.
