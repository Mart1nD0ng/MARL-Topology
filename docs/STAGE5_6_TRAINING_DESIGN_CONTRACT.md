# Stage 5.6 Training Design Contract Without Execution

## Controlled Object

The controlled object is the future decentralized MARL training workflow for
topology decisions:

```text
ActorObservation -> local edge actions -> topology evaluation
                 -> Stage 3/4 objective evidence
                 -> training-only surrogate diagnostics
                 -> future learner diagnostics
```

Stage 5.6 freezes the design contract. It does not execute training, implement
actor or critic models, write checkpoints, calibrate reward weights, select
final `tau_consensus`, or migrate v5 code.

## Desired State

The project has a concrete design boundary for future learning work while
training execution remains blocked.

Design verdict:

```text
design_contract_frozen_training_execution_blocked
training_execution_allowed = false
training_code_added = false
model_code_added = false
owner_decision_required = true
```

## Baseline

Stage 5.5 found that training execution was not ready because these gates were
missing:

- surrogate scalarization and weight policy;
- return, advantage, and value-target contract;
- actor and critic architecture contract;
- artifact, seed, and run-manifest policy;
- multi-seed stochastic evidence protocol.

Stage 5.6 resolves those missing items at the design-contract level only. It
does not implement them.

## Training Purpose

The future training purpose is:

```text
learn decentralized edge-activation behavior that satisfies the consensus
reliability requirement while reducing latency and energy
```

Deployment actors must remain Dec-POMDP compliant and use only local
observations, local history, and permitted local messages.

CTDE is allowed as a training paradigm, but Stage 5.6 does not commit to one
algorithm, one critic, or one policy architecture.

## Objective And Scalarization Policy

Evaluation objective remains:

```text
consensus_success_probability >= tau_requirement_min
then minimize latency and energy
```

`tau_requirement_min = 0.9` remains the requirement baseline. Lower tau values
are diagnostic only. Final `tau_consensus` selection still requires an owner
decision.

Training-side scalarization policy:

```text
constraint_first_component_policy_v1
```

Allowed future components:

- reliability violation penalty;
- normalized latency penalty;
- normalized energy penalty.

Required plateau:

```text
no reliability bonus above tau
```

Weight policy:

- no reward weights are selected in Stage 5.6;
- future weights must be explicit config fields;
- future weights must be justified by contract tests and report evidence;
- weights must not be copied from v5.

Forbidden scalarization routes:

- standalone timeout term;
- standalone quorum term;
- standalone density term;
- full graph bonus;
- oracle label target;
- old v5 formula or phase-script inheritance.

## Learning Target And Replay Contract

Currently admitted training-only diagnostic columns remain limited to Stage
5.2:

- `reward_surrogate`
- `reward_reliability_penalty`
- `reward_latency_penalty`
- `reward_energy_penalty`
- `reward_config_id`

Future learning columns require a later owner-approved replay-column contract
before they can appear in data:

- `return`
- `advantage`
- `value_target`
- `episode_id`
- `trajectory_id`
- `discount_factor`

Stage 5.6 does not add a dataset writer, replay buffer, learner batch, or
checkpoint loader.

## Actor And Critic Boundary

Allowed deployment actor fields remain:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

Forbidden deployment actor fields remain:

- global topology;
- oracle labels;
- registered evaluation metrics;
- surrogate diagnostics;
- future outcomes;
- centralized training views.

Future centralized critics may use training-only global state only after a
separate architecture contract. The future actor must still be loadable and
runnable without critic-only tensors.

## Scenario And Seed Protocol

Future training design must separate:

- training scenario set;
- validation scenario set;
- test scenario set;
- stress and boundary fixtures;
- deployment-calibration scenario set, if available.

Splits must be by `scenario_id` and seed, not by row shuffling alone.

Minimum future repeated-seed plan:

```text
at least five seeds before learning claims
```

Single-seed improvement, shaped-signal improvement, or full-graph reproduction
is diagnostic only.

## Baselines And Oracles

Required future comparison sensors:

- empty topology;
- sparse non-learning baseline;
- dense full-graph baseline;
- random local baseline;
- oracle candidate diagnostic.

Full graph remains a baseline, not an oracle. Oracle candidates remain
review-only diagnostics and must not enter actor inputs.

## Diagnostics

Registered evaluation evidence:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Training-only diagnostics may include:

- surrogate components;
- constraint violation rate;
- policy entropy;
- action sparsity;
- seed variance.

Training-only diagnostics do not replace registered evaluation metrics.

## Stop Conditions

Future execution contract must define:

- maximum update budget;
- validation plateau rule;
- constraint-failure stop rule;
- metric-regression stop rule;
- artifact budget rule.

Stage 5.6 defines these requirements only. It does not run or stop any process.

## Artifact Policy

Future artifacts must be written under `result_save` only after owner approval.

Each future run manifest must include:

- stage id;
- config id;
- scenario set id;
- seed;
- code version marker;
- contract ids;
- metric registry version.

Stage 5.6 writes no checkpoint, replay file, training log, or experiment
result.

## Acceptance Criteria

- Stage 5.6 design contract exists.
- Structured contract report is runnable.
- Training execution is explicitly disallowed.
- Scalarization policy exists without selected weights.
- Learning-target columns remain planned, not active.
- Deployment actor boundary is unchanged.
- Scenario/seed/baseline/artifact/stop-condition requirements are explicit.
- Tests and harness validation pass.

## Residual Risks

- Stage 5.6 is still design-only; it does not prove learning stability.
- Future architecture work must decide policy and critic structures without
  leaking global information into deployment actors.
- Deployment scenario distribution remains uncalibrated.
- Future execution still needs explicit owner approval.
