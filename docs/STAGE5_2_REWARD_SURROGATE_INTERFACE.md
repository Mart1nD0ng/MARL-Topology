# Stage 5.2 Reward Surrogate Interface Skeleton

## Scope

Stage 5.2 implements a minimal pure interface skeleton for a future
training-only reward surrogate. It does not train models, calibrate reward
weights, select final `tau_consensus`, add actor/critic/COMA/GNN/LSTM code, or
migrate v5 code.

Boundary shorthand: Stage 5.2 does not train and does not migrate v5 code.

The implementation file is:

- `src/marl_topology/objectives/surrogate_signal.py`

The file name avoids making a reward-named module the project entry point. The
public interface is still explicitly the Stage 5.2 reward-surrogate interface.

## Controlled Object

The controlled object is the boundary:

```text
registered Stage 3/4/5 evaluation quantities
-> training-only surrogate interface
-> decomposed training-side surrogate diagnostics
```

The controlled object is not a policy, actor, critic, replay writer, optimizer,
or scenario evaluator.

## Desired State

The project should have a narrow, testable interface that:

- consumes only registered evaluation concepts;
- treats consensus reliability as a constraint;
- plateaus reliability pressure above tau;
- penalizes latency and energy through explicit normalization references;
- returns decomposed training-only diagnostics;
- keeps surrogate fields out of deployment actor inputs;
- keeps reward values separate from registered evaluation metrics.

## Interface

Inputs:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Configuration:

- explicit `tau`;
- explicit nonnegative reliability, latency, and energy weights;
- explicit positive latency and energy references;
- explicit clipping bounds;
- explicit config id and model id.

Output record:

- scalar training signal value;
- reliability violation;
- reliability penalty;
- normalized latency;
- normalized energy;
- latency penalty;
- energy penalty;
- constraint-satisfied flag;
- config id;
- model id.

Training-only replay payload columns:

- `reward_surrogate`
- `reward_reliability_penalty`
- `reward_latency_penalty`
- `reward_energy_penalty`
- `reward_config_id`

These columns are not evaluation metrics and are forbidden deployment actor
inputs.

Rule shorthand: surrogate payload columns are forbidden deployment actor inputs.

## Formula

Stage 5.2 uses the Stage 5.1 shape with explicit config:

```text
reliability_violation = max(0, tau - consensus_success_probability)
reliability_penalty = reliability_violation ** reliability_penalty_power

latency_penalty = clip(latency / latency_reference_s)
energy_penalty = clip(energy / energy_reference_j)

training_signal = -(
    reliability_weight * reliability_penalty
    + latency_weight * latency_penalty
    + energy_weight * energy_penalty
)
```

If `consensus_success_probability >= tau`, reliability penalty is zero and no
reliability bonus is added.

Plateau shorthand:

```text
reliability_penalty = 0
no reliability bonus is added
```

## Sensors

Stage 5.2 sensors are:

- unit tests for plateau behavior, reliability violation, latency/energy
  monotonicity, invalid config fail-fast behavior, and replay boundary;
- contract tests for documentation, harness task, project state, source scans,
  and metric governance;
- full pytest;
- harness task validation;
- control-model rubric scoring.

## Actuators

Stage 5.2 changes only:

- objective interface module;
- replay column classification;
- contract documents;
- harness task;
- tests;
- project state;
- post-task self-review.

## Disturbances

Primary disturbances are:

- old v5 reward semantics leaking through names or aliases;
- treating surrogate values as evaluation metrics;
- adding default reward weights that look calibrated;
- allowing reward fields into deployment actor inputs;
- sliding into training or model implementation.

## Acceptance

Stage 5.2 is accepted only if:

- reward-surrogate inputs are registered metric concepts or declared
  diagnostics;
- reward-surrogate outputs are training-only columns, not metric names;
- deployment actor validation rejects surrogate columns;
- no training, optimizer, model, actor, critic, COMA, GNN, or LSTM code is
  added;
- no v5 code is migrated or modified;
- full graph remains a baseline, not an oracle;
- tests and harness validation pass.

## Deferred Work

Deferred to owner-approved future stages:

- reward normalization reference selection from an approved calibration source;
- reward weight calibration;
- reward report integration;
- return, advantage, and value-target contracts;
- training preflight review;
- actor/critic/model implementation;
- training runs.

## Residual Risks

- Stage 5.2 accepts explicit weights for testing and future config, but those
  weights are not calibrated or endorsed.
- The surrogate is scalar; future work must verify that scalarization does not
  hide reliability violations.
- Replay storage is still not implemented, so column-boundary protection is
  name-level rather than file-format-level.
