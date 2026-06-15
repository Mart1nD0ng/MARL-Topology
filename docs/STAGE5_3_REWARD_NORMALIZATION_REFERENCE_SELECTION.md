# Stage 5.3 Reward Normalization Reference Selection

## Scope

Stage 5.3 selects fixed latency and energy normalization references for the
Stage 5.2 training-only surrogate interface.

It does not calibrate reward weights, does not train, does not add
actor/critic/COMA/GNN/LSTM code, does not select final `tau_consensus`, and
does not migrate v5 code.

Boundary shorthand: Stage 5.3 does not train and does not migrate v5 code.

## Controlled Object

The controlled object is the reference boundary:

```text
approved Stage 3-backed objective evidence
-> fixed latency/energy normalization references
-> Stage 5.2 surrogate config inputs
```

The controlled object is not the reward scalarization, not a weight calibration
process, and not a training dataset.

## Source Evidence

Default source:

- Stage 5.0l Stage 3-backed range review
- source stage id:
  `stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review`
- executable sensor:
  `src/marl_topology/evaluation/feasibility_envelope_sweep_range_review.py`

The source rows use Stage 3 network communication records, Stage 3.6 URLLC
finite-blocklength links, Stage 4.3 message-matrix adaptation, Stage 4.6
protocol accounting, and Stage 4.4 expected-initiator PBFT reliability.

Stage 5.0l is still an alpha deterministic reference. These references are not
deployment calibration and do not claim physical realism beyond the documented
Stage 5.0l range review.

Rule shorthand: not deployment calibration.

## Selection Policy

Policy id:

```text
feasible_positive_max_v1
```

Eligibility:

```text
consensus_success_probability >= tau_requirement_min
latency > 0
energy > 0
tau_requirement_min = 0.9
```

Selection:

```text
latency_reference_s = max(latency over eligible rows)
energy_reference_j = max(energy over eligible rows)
```

Rationale:

- uses feasible objective-region rows only;
- avoids zero or failed rows as normalization anchors;
- keeps the first reference conservative and auditable;
- makes feasible Stage 5.0l rows normalize at or below 1;
- does not tune reward weights.

## Selected References

From the current Stage 5.0l source rows:

```text
source_row_count = 8
eligible_row_count = 4
excluded_row_count = 4
latency_reference_s = 0.0022698175688954207
energy_reference_j = 0.004134917967719052
```

Eligible latency source values:

```text
0.00180070349806436
0.001800738904705717
0.0022698175688954207
0.0022698175688954207
```

Eligible energy source values:

```text
0.003965692099788303
0.003971357232200643
0.004134917967719052
0.004134917967719052
```

## Implementation

Core selector:

- `src/marl_topology/objectives/normalization.py`

Report sensor:

- `src/marl_topology/evaluation/normalization_reference_selection.py`

Replay script:

- `scripts/replay/stage5_3_reward_normalization_reference_selection.py`

The selector can build a `SurrogateSignalConfig` with the selected references,
but caller-supplied weights remain explicit inputs. Stage 5.3 does not choose or
calibrate those weights.

## Metric Governance

Stage 5.3 does not add metric names.

Metric-valued source fields remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The normalization reference fields are configuration diagnostics, not
evaluation metrics.

## Dec-POMDP Boundary

Normalization references are training-side configuration. They are not
deployment actor observations and must not be inserted into `ActorObservation`
or deployment actor input batches.

## Negative Controls

Stage 5.3 preserves:

- no reward weight calibration;
- no training;
- no actor/critic/model implementation;
- no final `tau_consensus` selection;
- no v5 code migration;
- no old v5 reward or metric aliases;
- no oracle labels or calibration labels as actor inputs;
- no full graph as oracle.

## Acceptance

- Reference source is explicit and Stage 3-backed.
- Selection policy is fixed and deterministic.
- References are positive finite values.
- Eligible and excluded row counts are reported.
- The generated Stage 5.2 surrogate config uses the selected references.
- No new metrics, training, model code, reward weight calibration, final tau
  selection, or v5 migration occurs.

## Residual Risks

- The current references come from a small deterministic alpha suite, not a
  deployment-calibrated city distribution.
- Physical parameter realism remains limited by Stage 5.0l source ranges.
- Reward weights remain uncalibrated and should not be inferred from these
  references.
- Future larger scenario calibration may replace these references through an
  owner-approved update.

## Recommended Next Task

`Stage 5.4 - reward report integration without training`.

Reason: the pure interface and fixed references now exist. The next useful
sensor is to attach surrogate decomposition to evaluation reports as
training-only diagnostics, while still blocking training and weight calibration.
