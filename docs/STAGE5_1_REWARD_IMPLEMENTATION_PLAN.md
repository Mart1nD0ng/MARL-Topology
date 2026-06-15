# Stage 5.1 Reward Implementation Plan Without Code

## Scope

Stage 5.1 designs the future reward-surrogate implementation boundary. It does
not implement reward code, does not choose reward weights, does not train
models, does not add actor/critic/COMA/GNN/LSTM code, does not select final
`tau_consensus`, does not lower `tau_requirement_min = 0.9`, does not migrate
v5 code, and does not add reward columns to deployment actor inputs.

Boundary shorthand: no reward code in Stage 5.1.

## Controlled Object

The controlled object is the future training reward-surrogate interface for
MARL-Topology:

```text
Stage 3/4 evaluation records
-> future reward-surrogate adapter
-> training-only reward diagnostics
```

The controlled object is not the evaluation objective. Registered objective
metrics remain the source of truth.

## Desired State

The project should have a concrete implementation plan that a later owner-
approved stage can execute without re-opening old v5 reward mistakes.

Desired properties:

- reliability is a constraint;
- above-threshold reliability has a plateau by default;
- latency and energy are the optimization pressures;
- reward is a training surrogate, not a metric;
- all reward inputs come from registered evaluation quantities or declared
  diagnostics;
- actor observations remain Dec-POMDP-safe;
- reward, return, advantage, and value-target fields stay training-only;
- old v5 reward formulas, weights, aliases, and phase scripts remain rejected.

## Baseline Evidence

Stage 5.0m passed objective-readiness gates for a plan-only Stage 5.1 task.
The relevant evidence is:

- `tau_requirement_min = 0.9` is the current requirement baseline.
- Stage 5.0h classified infeasible rows by failure reason.
- Stage 5.0j showed alpha sweep recovery signals.
- Stage 5.0k grounded bandwidth, deadline, and resource orthogonalization in
  Stage 3-backed records.
- Stage 5.0l expanded Stage 3-backed coverage to tx power, payload, RSU
  height/placement, and resource-budget limits.

This evidence permits a plan. It does not by itself permit reward code,
training, or weight calibration.

## Proposed Future Module Boundary

Future implementation stage only, not Stage 5.1:

```text
src/marl_topology/objectives/reward_surrogate.py
```

The future module should expose a small pure function over explicit records.
It should not depend on actor models, critic models, training loops, replay
writers, or v5 artifacts.

Proposed future inputs:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics` only for disabled-by-default tie-breaker diagnostics
- declared `tau_requirement_min` or owner-approved `tau_consensus`
- declared normalization references

Proposed future outputs:

- scalar training surrogate value;
- reliability violation term;
- normalized latency term;
- normalized energy term;
- constraint-satisfied flag;
- diagnostic decomposition for reports;
- config id.

These outputs are not evaluation metrics. If they enter replay, they must be
classified as training-only or evaluation-only columns before use.

## Surrogate Shape Plan

The future surrogate should use this shape:

```text
reliability_violation = max(0, tau - consensus_success_probability)
reliability_penalty = penalty_fn(reliability_violation)

latency_penalty = normalize_latency(latency)
energy_penalty = normalize_energy(energy)

reward_surrogate = - reliability_weight * reliability_penalty
                   - latency_weight * latency_penalty
                   - energy_weight * energy_penalty
                   + optional_disabled_tie_breaker
```

Stage 5.1 does not choose `reliability_weight`, `latency_weight`, or
`energy_weight`. A future implementation stage must require them in config and
must test that missing weights fail fast.

Reliability plateau rule:

```text
if consensus_success_probability >= tau:
    reliability_penalty = 0
    no reliability bonus is added by default
```

This prevents redundant topology pressure after the reliability constraint is
satisfied.

## Tau Policy

The future reward surrogate must not fit tau from the current alpha fixtures.

Current planning rule:

```text
tau_requirement_min = 0.9
```

Future implementation may accept `tau` through config. It must not hardcode a
lower value and must not treat lower diagnostic values from Stage 5.0h as final
thresholds.

Final `tau_consensus` selection remains owner-controlled.

## Normalization Plan

Future implementation must require fixed, explicit normalization references:

| Quantity | Required reference | Stage 5.1 decision |
| --- | --- | --- |
| reliability violation | bounded by `[0, 1]` | no extra scale selected |
| latency | positive `latency_reference_s` | must be supplied by config |
| energy | positive `energy_reference_j` | must be supplied by config |
| clipping | explicit min/max policy | must be supplied by config |
| missing records | explicit failure policy | must be supplied by config |

Default guidance:

- Prefer fixed references from an owner-approved calibration set.
- Do not use batch-local normalization for first implementation because it can
  hide constraint violations and make rewards non-comparable across scenarios.
- Do not infer references from the current batch during deployment evaluation.

## Forbidden Components

The future reward implementation must not include:

- standalone timeout reward;
- standalone quorum reward;
- standalone deadline reward;
- standalone edge-count reward;
- standalone density or sparsity reward;
- full graph or full mask bonus;
- reliability bonus above tau by default;
- old v5 reward formulas;
- old v5 weights or sentinel penalties;
- `P_eff` aliases or hard/soft legacy names;
- oracle labels as actor inputs or reward labels;
- global topology as deployment actor input.

Timeout, quorum, and deadline may affect registered reliability, latency, or
energy through Stage 3/4 records. They are not reward terms by name.

## Dec-POMDP Boundary

Reward is computed environment-side or training-side after topology evaluation.
It is not a deployment actor observation.

Actor inputs must remain a subset of `ActorObservation`:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Forbidden actor inputs include reward, return, advantage, value target,
consensus success, latency, energy, oracle labels, full topology, future
outcomes, and centralized critic features.

## Replay Dataset Plan

Stage 5.1 does not admit new replay columns yet. Future replay contract update
must classify:

- `reward_surrogate`: training-only;
- `reward_reliability_penalty`: training-only diagnostic;
- `reward_latency_penalty`: training-only diagnostic;
- `reward_energy_penalty`: training-only diagnostic;
- `reward_config_id`: training-only or evaluation-only identifier;
- `return`, `advantage`, and `value_target`: unsupported until training
  contracts admit them.

None of these columns may enter deployment actor input projections.

## Required Future Tests

Before reward implementation can be accepted, tests must cover:

- unreliable low-latency topology is penalized below a reliable higher-latency
  topology;
- above-tau reliability receives no extra reward bonus by default;
- lower latency improves reward only among equally feasible rows or through the
  declared scalarization;
- lower energy improves reward only among equally feasible rows or through the
  declared scalarization;
- edge count, density, full graph, timeout, quorum, and deadline do not appear
  as active reward components;
- all reward inputs map to registered metrics or declared diagnostics;
- missing normalization references fail fast;
- reward values do not enter actor observations;
- replay actor projection excludes reward/return/advantage/value target fields;
- old v5 reward names and metric aliases are absent from source.

## Future Implementation Sequence

Future implementation sequence:

Recommended future stages:

1. `Stage 5.2 - reward surrogate interface skeleton with contract tests`
   Implement only a pure reward-surrogate adapter and tests. No training.
2. `Stage 5.3 - reward normalization reference selection`
   Choose or load fixed latency/energy references from an approved calibration
   set. No training.
3. `Stage 5.4 - reward report integration`
   Add reward decomposition to evaluation reports as training diagnostics only.
4. `Stage 5.5 - training preflight review`
   Re-check Dec-POMDP, replay columns, baselines, reward-hacking tests, and
   metric reporting before any training run.

## Acceptance For Stage 5.1

- Plan states reward as a training surrogate, not an evaluation metric.
- Plan preserves reliability as a constraint and latency/energy as objectives.
- Plan includes reliability plateau above tau.
- Plan requires explicit normalization references and weights before code.
- Plan blocks reward code, reward weights, training, actor/critic/model work,
  final tau selection, and v5 migration.
- Plan includes Dec-POMDP and replay-column leakage guards.
- Harness and contract tests protect the plan-only boundary.

## Residual Risks

- The first implementation stage can still create reward hacking if it skips
  negative tests.
- Latency and energy scalarization remains intentionally unresolved until an
  owner-approved implementation config.
- Physical parameter realism and scenario distribution calibration are still
  limited relative to deployment claims.
