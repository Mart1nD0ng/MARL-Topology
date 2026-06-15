# Stage 5.0 Reward / Objective Contract Freeze

## Controlled Object

The controlled object is the Stage 5 objective and reward-surrogate contract
boundary for MARL-Topology.

Stage 5.0 freezes semantics only. It does not implement reward, train models,
add actor/critic/COMA/GNN/LSTM code, tune weights, or migrate v5 code.

Boundary shorthand: Stage 5.0 does not train models.

## Desired State

The project should be ready to discuss future reward implementation without
semantic ambiguity:

- objective and future reward surrogate are separate;
- consensus reliability is a feasibility constraint;
- latency and energy are primary optimization objectives;
- diagnostics remain diagnostics or explicit tie-breakers;
- Stage 4.8 audit threshold remains a diagnostic value, not final
  `tau_consensus`;
- reward implementation and training remain blocked.

## Frozen Semantics

### Objective

The evaluation objective is:

```text
feasible if consensus_success_probability >= tau_consensus
then minimize latency and energy among feasible topologies
```

`tau_consensus` is an objective-contract parameter. Stage 5.0 does not choose
its value.

### Future Training Surrogate

A future surrogate may use a smooth penalty below `tau_consensus` plus
normalized latency and energy penalties. Above `tau_consensus`, reliability
plateaus by default and must not produce an unlimited bonus.

The surrogate is not a metric and cannot be reported as success without
registered metric evidence.

## Stage 4.8 Boundary Use

Stage 4.8 provided a deterministic boundary audit with non-saturated,
failed-scheduled-message, unreachable-target, weak-primary, center-primary,
full-graph-penalty, and sparse-resource cases.

Its `reliability_threshold = 0.2` is only an audit diagnostic threshold. It is
not `tau_consensus`.

## Deferred Work

- selecting or calibrating `tau_consensus`;
- reward implementation;
- reward weight calibration;
- reward-hacking test implementation beyond contract checks;
- training runs;
- actor, critic, COMA, GNN, or LSTM implementation;
- large scenario calibration;
- replay reward/return dataset columns.

## Acceptance

- `docs/OBJECTIVE_CONTRACT.md` exists and defines objective semantics.
- `docs/REWARD_SURROGATE_CONTRACT.md` exists and defines future surrogate
  requirements.
- `docs/REWARD_CONTRACT.md` records Stage 5.0 without activating reward code.
- Harness has a Stage 5.0 task.
- Tests verify no reward implementation, training, or model code was added.
- v5 remains read-only and no v5 reward logic is inherited.

## Residual Risks

- The actual `tau_consensus` still needs scenario calibration or owner
  decision.
- Latency/energy scalarization remains unresolved by design.
- A future reward implementation may still create reward hacking unless it
  adds negative tests before training.

## Stage 5.0a Follow-Up

Stage 5.0a is documented in:

`docs/TAU_CONSENSUS_CALIBRATION_PLAN.md`

It defines the calibration evidence and owner-decision process for
`tau_consensus`. It still does not select the final threshold, implement reward,
calibrate reward weights, or run training.

Stage 5.0c is documented in:

`docs/TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md`

It defines future report tables, fixture families, diagnostics, and validation
gates. It still does not run calibration or select the final threshold.

Stage 5.0d is documented in:

`docs/STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md`

It implements a report-only sensor that requires explicit candidate tau values.
It still does not select `tau_consensus`, implement reward, calibrate reward
weights, or run training.

Stage 5.0e is documented in:

`docs/STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md`

It defines the scenario-family coverage required before tau calibration can be
run as more than a smoke test. It still does not implement fixtures, run
calibration, or select the final threshold.

Stage 5.0f is documented in:

`docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`

It implements a minimal executable alpha fixture suite and report-source
adapter. It still does not run final tau selection, implement reward, calibrate
reward weights, or train models.
