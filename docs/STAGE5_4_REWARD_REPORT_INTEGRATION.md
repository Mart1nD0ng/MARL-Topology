# Stage 5.4 Reward Report Integration Without Training

## Scope

Stage 5.4 attaches Stage 5.2 surrogate decomposition and Stage 5.3
normalization references to an evaluation report as training-only diagnostics.

It does not report scalar surrogate as an evaluation metric, does not calibrate
reward weights, does not train, does not add actor/critic/COMA/GNN/LSTM code,
does not select final `tau_consensus`, and does not migrate v5 code.

Boundary shorthand: Stage 5.4 does not train and does not migrate v5 code.
Boundary shorthand: Stage 5.4 does not calibrate reward weights.

## Controlled Object

The controlled object is:

```text
Stage 5.0l objective rows
-> Stage 5.3 normalization references
-> Stage 5.2 surrogate decomposition
-> training-only diagnostic report
```

The controlled object is not an actor observation, not a metric table, not a
training loop, and not reward-weight calibration.

## Integration Policy

Policy id:

```text
component_only_no_scalar_reward_v1
```

The report computes and shows:

- reliability violation;
- reliability penalty;
- normalized latency;
- normalized energy;
- latency penalty;
- energy penalty;
- constraint-satisfied flag;
- normalization reference provenance.

The report intentionally omits scalar training signal values:

```text
scalar_surrogate_reported = false
```

This prevents uncalibrated weights from becoming a hidden training objective or
a misleading evaluation claim.

## Source Evidence

Source rows:

- Stage 5.0l Stage 3-backed range review rows.

Reference source:

- Stage 5.3 normalization references.

Selected Stage 5.3 normalization references:

```text
latency_reference_s = 0.0022698175688954207
energy_reference_j = 0.004134917967719052
```

## Implementation

Module:

- `src/marl_topology/evaluation/surrogate_diagnostics_report.py`

Replay script:

- `scripts/replay/stage5_4_reward_report_integration.py`

Report stage id:

```text
stage_5_4_reward_report_integration_without_training
```

## Report Fields

Each surrogate diagnostic row contains:

- source row id;
- source scenario family;
- topology name;
- `consensus_success_probability`;
- `latency`;
- `energy`;
- reliability violation and penalty;
- normalized latency and energy;
- latency and energy penalties;
- tau;
- normalization references;
- training-only boundary flags.

These fields are training-only diagnostics, not evaluation metrics.

## Metric Governance

Stage 5.4 does not add metric names.

Metric-valued inputs remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Surrogate decomposition fields, normalization references, component-policy ids,
and report-stage ids are diagnostics or identifiers. They must not be exported
as metric rows.

## Dec-POMDP Boundary

Surrogate diagnostic rows are not deployment actor inputs. They must not enter
`ActorObservation`, deployment actor projection, policy checkpoints, or
deployment replay columns.

## Acceptance

- Report rows match the Stage 5.0l source row count.
- Stage 5.3 normalization references are attached.
- Constraint violation and plateau rows are both visible.
- Normalized latency and energy are positive and auditable.
- All surrogate outputs are training-only diagnostics.
- No scalar surrogate is reported.
- No new metrics, training, model code, reward weight calibration, final tau
  selection, or v5 migration occurs.

## Residual Risks

- Component-only diagnostics still come from the small deterministic Stage 5.0l
  source, not a deployment-calibrated city distribution.
- Future report consumers could overinterpret normalized penalties unless raw
  registered metrics remain visible.
- The next stage must review training preconditions before any training run or
  weight calibration.

## Recommended Next Task

`Stage 5.5 - training preflight review without training`.

Reason: surrogate interface, fixed references, and report diagnostics now
exist. Before training, the project needs a preflight review covering Dec-POMDP
leakage, replay columns, reward-hacking tests, baselines, report semantics, and
blocked weight calibration.
